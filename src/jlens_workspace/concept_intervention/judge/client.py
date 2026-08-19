"""Minimal secret-safe OpenRouter client with strict structured outputs."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from jlens_workspace.concept_intervention.judge.prompts import (
    render_messages,
    response_schema,
    validate_judgment,
)
from jlens_workspace.concept_intervention.judge.three_method_config import OpenRouterConfig


class OpenRouterError(RuntimeError):
    """Raised when a judge request cannot produce a registered valid response."""

    def __init__(
        self,
        message: str,
        *,
        attempt_records: tuple[dict[str, Any], ...] = (),
    ) -> None:
        super().__init__(message)
        self.attempt_records = attempt_records


class OpenRouterContentFilterError(OpenRouterError):
    """Every registered provider attempt returned a content-filter outcome."""


class _InvalidJudgeResponse(ValueError):
    """A completed provider response that is safe to retry without changing the task."""


@dataclass(frozen=True)
class JudgeResponse:
    """One validated response plus auditable non-secret routing metadata."""

    requested_model: str
    returned_model: str
    response_id: str
    provider: str | None
    created: int | None
    finish_reason: str | None
    usage: dict[str, Any]
    judgment: dict[str, Any]
    attempts: int
    validation_warnings: tuple[str, ...] = ()
    request_payload: dict[str, Any] = field(default_factory=dict)
    raw_provider_response: dict[str, Any] = field(default_factory=dict)
    attempt_records: tuple[dict[str, Any], ...] = ()
    request_started_at: str | None = None
    response_received_at: str | None = None


Transport = Callable[[str, Mapping[str, str], bytes, float], Mapping[str, Any]]
KeyTransport = Callable[[str, Mapping[str, str], float], Mapping[str, Any]]


def _urllib_transport(
    endpoint: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise OpenRouterError("OpenRouter returned a non-object response")
    return payload


def _urllib_key_transport(
    endpoint: str, headers: Mapping[str, str], timeout: float
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        endpoint,
        headers=dict(headers),
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise OpenRouterError("OpenRouter key endpoint returned a non-object response")
    return payload


def _recover_truncated_pairwise_judgment(content: str) -> dict[str, Any]:
    """Recover only complete, unique decision fields from length-truncated JSON."""

    if not content.lstrip().startswith("{"):
        raise _InvalidJudgeResponse("truncated pairwise response is not a JSON object")
    specifications = {
        "target_preference": r'"target_preference"\s*:\s*"(A|B|tie)"',
        "quality_preference": r'"quality_preference"\s*:\s*"(A|B|tie)"',
        "target_strength": r'"target_strength"\s*:\s*"(none|small|moderate|large)"',
        "confidence": r'"confidence"\s*:\s*(-?\d+)(?=\s*[,}])',
    }
    recovered: dict[str, Any] = {}
    positions: list[int] = []
    for schema_field, pattern in specifications.items():
        if len(re.findall(rf'"{schema_field}"\s*:', content)) != 1:
            raise _InvalidJudgeResponse(
                f"truncated pairwise response lacks one unique {schema_field} field"
            )
        matches = list(re.finditer(pattern, content))
        if len(matches) != 1:
            raise _InvalidJudgeResponse(
                f"truncated pairwise response has no unambiguous {schema_field} value"
            )
        match = matches[0]
        positions.append(match.start())
        value: Any = match.group(1)
        recovered[schema_field] = (
            int(value) if schema_field == "confidence" else value
        )
    if positions != sorted(positions):
        raise _InvalidJudgeResponse(
            "truncated pairwise decision fields are outside registered schema order"
        )
    return {
        **recovered,
        "evidence_a": [],
        "evidence_b": [],
        "rationale": "",
    }


class OpenRouterClient:
    """Call one exact judge model without cross-model or data-collection fallback."""

    def __init__(
        self,
        config: OpenRouterConfig,
        *,
        transport: Transport | None = None,
        key_transport: KeyTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transport = transport or _urllib_transport
        self.key_transport = key_transport or _urllib_key_transport
        self.sleeper = sleeper

    def key_status(self) -> dict[str, Any]:
        """Read the authenticated key's live limits without sending task content."""

        key = os.environ.get(self.config.api_key_env, "").strip()
        if not key:
            raise OpenRouterError(
                f"missing API key environment variable {self.config.api_key_env}"
            )
        headers = {
            "Authorization": f"Bearer {key}",
            "User-Agent": "jlens-workspace-judge/1",
        }
        try:
            payload = self.key_transport(
                self.config.key_status_endpoint,
                headers,
                self.config.timeout_seconds,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            detail = (
                f"HTTP {error.code}"
                if isinstance(error, urllib.error.HTTPError)
                else type(error).__name__
            )
            raise OpenRouterError(
                f"OpenRouter key-status check failed: {detail}"
            ) from error
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise OpenRouterError("OpenRouter key-status response lacks a data object")
        return {str(name): value for name, value in data.items()}

    def judge(self, *, model: str, task: Mapping[str, Any]) -> JudgeResponse:
        """Request, parse, and semantically validate one blind task."""

        key = os.environ.get(self.config.api_key_env, "").strip()
        if not key:
            raise OpenRouterError(
                f"missing API key environment variable {self.config.api_key_env}"
            )
        task_type = str(task.get("task_type"))
        if task_type not in {"pointwise", "pairwise"}:
            raise OpenRouterError(f"unsupported task type: {task_type!r}")
        request_payload: dict[str, Any] = {
            "model": model,
            "messages": render_messages(task),
            "max_tokens": self.config.max_tokens,
            "response_format": response_schema(task_type),
        }
        if model not in self.config.temperature_unsupported_models:
            request_payload["temperature"] = self.config.temperature
        reasoning_effort = self.config.reasoning_effort_by_model.get(model)
        if reasoning_effort is not None:
            request_payload["reasoning"] = {
                "effort": reasoning_effort,
                "exclude": self.config.exclude_reasoning,
            }
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "jlens-workspace-judge/1",
        }
        last_error: BaseException | None = None
        attempts = self.config.max_retries + 1
        provider_order = self.config.provider_order_by_model.get(model, [])
        content_filter_attempts = 0
        attempt_records: list[dict[str, Any]] = []
        for attempt in range(1, attempts + 1):
            provider_policy: dict[str, Any] = {
                "data_collection": self.config.data_collection,
                "require_parameters": self.config.require_parameters,
                "allow_fallbacks": self.config.allow_provider_fallbacks,
            }
            if provider_order:
                offset = (attempt - 1) % len(provider_order)
                provider_policy["order"] = (
                    provider_order[offset:] + provider_order[:offset]
                )
            request_payload["provider"] = provider_policy
            body = json.dumps(request_payload, separators=(",", ":")).encode("utf-8")
            request_started_at = datetime.now(UTC).isoformat()
            attempt_record: dict[str, Any] = {
                "attempt": attempt,
                "request_started_at": request_started_at,
                "provider_order": list(provider_policy.get("order", [])),
                "request_payload": json.loads(body.decode("utf-8")),
            }
            payload: Mapping[str, Any] | None = None
            try:
                payload = self.transport(
                    self.config.endpoint,
                    headers,
                    body,
                    self.config.timeout_seconds,
                )
                response_received_at = datetime.now(UTC).isoformat()
                response = self._parse_response(
                    payload, model=model, task=task, attempts=attempt
                )
                attempt_record.update(
                    {
                        "response_received_at": response_received_at,
                        "outcome": "validated",
                        "response_id": response.response_id,
                        "provider": response.provider,
                        "returned_model": response.returned_model,
                        "finish_reason": response.finish_reason,
                    }
                )
                attempt_records.append(attempt_record)
                return replace(
                    response,
                    request_payload=json.loads(body.decode("utf-8")),
                    raw_provider_response=dict(payload),
                    attempt_records=tuple(attempt_records),
                    request_started_at=request_started_at,
                    response_received_at=response_received_at,
                )
            except OpenRouterError as error:
                attempt_record.update(
                    {
                        "response_received_at": datetime.now(UTC).isoformat(),
                        "outcome": "registration_error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                if payload is not None:
                    attempt_record["raw_provider_response"] = dict(payload)
                attempt_records.append(attempt_record)
                raise OpenRouterError(
                    str(error),
                    attempt_records=tuple(attempt_records),
                ) from error
            except urllib.error.HTTPError as error:
                last_error = error
                attempt_record.update(
                    {
                        "response_received_at": datetime.now(UTC).isoformat(),
                        "outcome": "http_error",
                        "error_type": type(error).__name__,
                        "http_status": error.code,
                    }
                )
                attempt_records.append(attempt_record)
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt == attempts:
                    break
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                attempt_record.update(
                    {
                        "response_received_at": datetime.now(UTC).isoformat(),
                        "outcome": "transport_error",
                        "error_type": type(error).__name__,
                    }
                )
                attempt_records.append(attempt_record)
                if attempt == attempts:
                    break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                last_error = error
                content_filtered = "finish_reason='content_filter'" in str(error)
                attempt_record.update(
                    {
                        "response_received_at": datetime.now(UTC).isoformat(),
                        "outcome": (
                            "content_filter" if content_filtered else "validation_error"
                        ),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                if payload is not None:
                    attempt_record["raw_provider_response"] = dict(payload)
                attempt_records.append(attempt_record)
                if content_filtered:
                    content_filter_attempts += 1
                if attempt == attempts:
                    if content_filter_attempts == attempts:
                        raise OpenRouterContentFilterError(
                            "all registered provider attempts returned content_filter: "
                            f"{attempts} attempt(s)",
                            attempt_records=tuple(attempt_records),
                        ) from error
                    raise OpenRouterError(
                        "judge response failed registered validation after "
                        f"{attempt} attempt(s): {error}",
                        attempt_records=tuple(attempt_records),
                    ) from error
            self.sleeper(min(30.0, 2.0 ** (attempt - 1)))
        if isinstance(last_error, urllib.error.HTTPError):
            detail = f"HTTP {last_error.code}"
        else:
            detail = type(last_error).__name__ if last_error else "unknown error"
        raise OpenRouterError(
            f"OpenRouter request failed after {attempt} attempt(s): {detail}",
            attempt_records=tuple(attempt_records),
        ) from last_error

    @staticmethod
    def _parse_response(
        payload: Mapping[str, Any],
        *,
        model: str,
        task: Mapping[str, Any],
        attempts: int,
    ) -> JudgeResponse:
        if payload.get("error"):
            raise OpenRouterError("OpenRouter returned an error object")
        returned_model = str(payload.get("model", ""))
        if returned_model != model:
            raise OpenRouterError(
                f"resolved model differs from registration: {returned_model!r} != {model!r}"
            )
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise _InvalidJudgeResponse(
                "judge response must contain exactly one choice"
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise _InvalidJudgeResponse("malformed judge choice")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise _InvalidJudgeResponse("judge choice lacks a message")
        content = message.get("content")
        if not isinstance(content, str):
            finish_reason = choice.get("finish_reason")
            reasoning_present = bool(
                message.get("reasoning") or message.get("reasoning_details")
            )
            raise _InvalidJudgeResponse(
                "judge message content is not a JSON string "
                f"(finish_reason={finish_reason!r}, reasoning_present={reasoning_present})"
            )
        validation_warnings: list[str] = []
        try:
            judgment_payload = json.loads(content)
        except json.JSONDecodeError:
            finish_reason = str(choice.get("finish_reason", ""))
            if (
                task.get("task_type") != "pairwise"
                or finish_reason not in {"length", "max_tokens"}
            ):
                raise
            judgment_payload = _recover_truncated_pairwise_judgment(content)
            validation_warnings.append(
                "pairwise: recovered complete decision fields from length-truncated JSON; "
                "omitted evidence and rationale"
            )
        if not isinstance(judgment_payload, Mapping):
            raise _InvalidJudgeResponse("structured judgment is not an object")
        judgment = validate_judgment(
            task, judgment_payload, warnings=validation_warnings
        )
        usage = payload.get("usage", {})
        if not isinstance(usage, Mapping):
            usage = {}
        created = payload.get("created")
        return JudgeResponse(
            requested_model=model,
            returned_model=returned_model,
            response_id=str(payload.get("id", "")),
            provider=(
                str(payload["provider"])
                if payload.get("provider") is not None
                else None
            ),
            created=int(created) if isinstance(created, int) else None,
            finish_reason=(
                str(choice["finish_reason"])
                if choice.get("finish_reason") is not None
                else None
            ),
            usage={str(key): value for key, value in usage.items()},
            judgment=judgment,
            attempts=attempts,
            validation_warnings=tuple(validation_warnings),
        )
