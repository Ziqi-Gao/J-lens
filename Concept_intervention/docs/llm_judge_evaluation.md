# Blinded J-component/RAPTOR LLM-judge protocol

## Status and scope

This is the registered evaluation lane for
`qwen35_4b_j_raptor_llm_judge_pilot_v8`. It evaluates completed J-component and
RAPTOR generations without waiting for ITI. It does not mutate or reinterpret
the three-method experiment and it does not treat LLM judgments as ground
truth.

The registration is
[`qwen35_4b_j_raptor_llm_judge_pilot_v8.yaml`](../configs/qwen35_4b_j_raptor_llm_judge_pilot_v8.yaml).
Its prepared artifacts live in a new `llm_judge_pilot_v8/` child beside the
method outputs. API credentials are never stored in the YAML, task files,
response files, or logs.

Recovery v8 supersedes, but does not overwrite, the v1--v7 attempts. V1 stopped
at 362/672 primary pointwise-validation results after three zero-progress
batches: the model catalog showed optional adaptive reasoning enabled at high
effort by default for Sonnet 5, while the request allowed only 400 completion
tokens. V2 then stopped at 17/672 after Amazon Bedrock repeatedly returned
`content_filter` for two blind tasks, including the same task across independent
controller batches. Retrying that endpoint would spend budget without making
progress. V3 registered native Anthropic first, but native Anthropic and Bedrock
both filtered the same sample; it was stopped at 9/672, showing that the failure
was model-level rather than endpoint-level. V4 was prepared locally but sent no
remote request because xAI was outside the operator's explicit destination
authorization.

V5 kept every sample and moved the base estimator to Google Gemini 3.5 Flash
and Anthropic Claude Haiku 4.5. Its Haiku smoke succeeded, but a formal batch
accepted only 9/10 tasks because otherwise valid score JSON twice included a
paraphrased evidence string. Evidence is audit metadata, not part of either
estimator; discarding an entire score for that field would create avoidable
missingness and repeat spend. V6 kept the same approved providers, models,
blinded task construction, score anchors, and gates. Rubric v2 made evidence
explicitly optional and deterministically dropped only non-exact or
out-of-bounds evidence items. Its Haiku smoke succeeded, but the first formal
batch again accepted 9/10 because another valid score JSON twice exceeded the
600-character rationale limit. V7 generalized the same rule to all non-scoring
audit text: rubric v3 explicitly limits rationale length, the validator truncates
only excess rationale characters and records a warning, and evidence sanitation
is unchanged. Scores, preferences, and flags remain fail-closed and unmodified.
V7 recovery tests then completed 20/20 tasks for each base model with no failure;
the full controller stopped at 270/672 primary pointwise tasks before transmission
because one conservative UTF-8-byte prompt estimate was 5,005 against the
registered 5,000 limit. An offline audit of all 1,792 public tasks found a maximum
estimate of 5,085. V8 keeps the exact models, rubric, tasks, and scoring protocol,
raising only that per-request preflight bound to 5,120 while retaining the
275,000-token invocation limit and all dollar gates. It may reuse already completed
v7 responses only after task-file, task, rubric, model, and protocol hashes match
exactly. A migration manifest records every imported response hash, so judgments
from materially different registrations are never mixed.

The current prepared pilot contains:

| Task set | Per split | Purpose |
|---|---:|---|
| Pointwise | 672 | 7 concepts x 16 open prompts x 6 conditions x greedy |
| Pairwise | 224 | 7 concepts x 16 open prompts x J/RAPTOR x both A/B orders |

The six pointwise conditions are the common zero/no-hook baseline, full probe,
J component, non-J component, one matched random control, and RAPTOR. The
pairwise task compares only selected J and RAPTOR generations. A copied future
registration can add all three sample seeds and all five random controls; the
pilot deliberately starts with greedy outputs and one random control.

## Research basis

The protocol combines several established practices rather than copying one
benchmark prompt verbatim:

- [Inference-Time Intervention (ITI)](https://proceedings.neurips.cc/paper_files/paper/2023/file/81b8390039b7302c909cb769f8b6cd93-Paper-Conference.pdf)
  tunes intervention choices on validation data, evaluates a held-out split,
  reports desired behavior separately from informativeness, and checks the
  automatic judges against blinded humans. Here, candidate-label validation
  selects intervention strength/probability, while open test prompts remain
  untouched until the judge calibration gate passes.
- [Anthropic Persona Vectors](https://www.anthropic.com/research/persona-vectors)
  uses trait-specific 0--100 evaluation and a separate coherence judge. Its
  [reference implementation](https://github.com/safety-research/persona_vectors)
  informed the target-behavior anchors and the separation of target expression
  from coherence.
- [Anthropic Bloom](https://www.anthropic.com/research/bloom) asks a judge to
  use a detailed behavior definition, structured scores, a concise rationale,
  and transcript evidence. Its
  [reference implementation](https://github.com/safety-research/bloom)
  informed the exact-evidence fields and structured JSON response.
- [Judging LLM-as-a-Judge](https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf)
  documents position, verbosity, and self-enhancement biases. Every J/RAPTOR
  comparison is therefore judged in both response orders, and the position
  inconsistency rate is reported rather than silently discarded.

The resulting prompt is versioned as
`concept_intervention_judge_rubric_v3`; its complete text and strict JSON
schemas are in `src/jlens_workspace/concept_intervention/judge/prompts.py`.
The prompt/schema hash is frozen in the preparation manifest and every result.
The wire schemas use the strict subset shared by the registered providers.
Scores, preferences, and flags are enforced locally and fail closed. Non-scoring
audit text is post-validated separately: only exact contiguous evidence
substrings of the registered response, within the count and length bounds, are
retained, and rationales longer than 600 characters are deterministically
truncated to that limit. Every change is reported in the result's
`validation_warnings`; scored fields never change. This split follows
Anthropic's structured-output guidance to remove unsupported schema constraints
(such as `minimum`, `maximum`, `minLength`, and `maxLength`) and validate them in
application code.

## Registered judges

No OpenAI model, automatic router, free route, or `:latest` alias is allowed.
Every role is a distinct exact OpenRouter model ID:

| Role | Model | Use in the estimator |
|---|---|---|
| Primary | `google/gemini-3.5-flash` | yes |
| Secondary | `anthropic/claude-haiku-4.5` | yes |
| Arbitration | `google/gemini-3.6-flash` | disagreement sensitivity only |
| Expert review | `anthropic/claude-opus-5` | unresolved/random audit only |

The primary pointwise score is the unweighted mean of primary and secondary
judges. Selective arbitration and expert review never replace that value; they
are sensitivity/audit outputs. This prevents cases selected for disagreement
from receiving a different estimator.

OpenRouter requests set temperature zero, require strict structured-output
support, deny provider data collection, and disable within-request provider
fallback. The returned model ID must equal the requested model ID exactly;
cross-model fallback remains prohibited. For Anthropic roles, an application
retry rotates from Anthropic's native endpoint to Amazon Bedrock while keeping
the exact registered model fixed. Per-task response files make the run
resumable without appending partial JSONL. Recovery v8 uses `minimal` reasoning
for Gemini 3.5/3.6 Flash and `low` for Claude Opus 5. Claude Haiku 4.5 receives
no reasoning-effort parameter because the current catalog does not support it.
These settings leave room inside the 1,024-token completion limit for the
strict JSON answer. Reasoning traces are excluded from the returned response
and are never stored.

## Blinding and rubrics

Judges see the target concept name/definition, user prompt, and one or two
responses. They do not see method, condition, strength/probability, generation
decoding, seed, source filename, or hypothesis. Public task order is
deterministically shuffled. Private maps are mode `0600` under a mode `0700`
directory.

Pointwise judgments return:

- target expression, 0--100;
- coherence, 0--100;
- relevance, 0--100;
- refusal and invalid flags;
- up to three exact response substrings and a short rationale.

Pairwise judgments return target preference and quality preference separately,
plus target-effect strength, confidence, exact evidence, and rationale. A
longer answer is explicitly not preferred merely for length. Prompt/response
content is delimited as quoted data to reduce prompt-injection risk.

There is no multiplicative or opaque composite primary score. Target behavior,
quality, and refusal are reported separately, analogous to ITI's reason for
not treating a trivial refusal as successful behavior.

## Selection and inference

For each concept, J strength is the positive `j` strength with the largest
mean candidate-label target margin on the candidate validation prompts; ties
choose the smaller strength. RAPTOR probability is selected from values above
0.5 with the same validation metric; ties choose the probability closest to
0.5. The selected J strength is then applied to full, J, non-J, and random
controls. Open test judgments play no role in selection.

The primary estimand is the equal-concept macro mean of the paired difference

`J target-expression score - RAPTOR target-expression score`

on open test prompts. Pointwise outcomes are paired on concept, prompt ID,
decoding, and seed. Bootstrap and sign-randomization operate on prompt-ID
clusters, so multiple decodings of a prompt never become independent samples.
Concepts receive equal macro weight.

Quality is a guardrail, not part of the target score. The registered checks are
a lower 95% confidence bound of at least -10 points for mean
coherence/relevance and an upper 95% bound of at most +0.05 for refusal rate.
Full, non-J, random, and baseline contrasts are secondary; their target-score
p-values use Benjamini--Hochberg correction. Pairwise results report stable J,
RAPTOR, tie, and discordant counts, plus order consistency and position-bias
rate.

## Calibration gate

Both base judges must first complete all validation pointwise and pairwise
tasks. Test access remains locked unless all registered gates pass:

- at least 98% completion;
- pointwise target-expression Spearman correlation at least 0.60;
- pointwise mean absolute difference at most 20 points;
- at least 80% A/B order consistency for each judge;
- at least 70% agreement between judges on stable pairwise preferences.

The internal pilot explicitly permits this machine-only gate. A
publication-grade copied registration should set `require_human_gate: true`,
freeze independently double-annotated human labels, and require human/judge
agreement before unlocking test. Never tune the rubric or its thresholds after
looking at test judgments.
The exported pilot audit is stratified within every concept: four pointwise and
four pairwise tasks, for 56 blinded tasks total.


## External-data warning

Running `judge prepare`, `judge validate`, and the offline tests is local.
Running `judge run` sends the public blind task's real project prompt, concept
definition, and generated response to OpenRouter and the selected provider.
Blinding hides experimental identity but does not anonymize the text itself.
Obtain explicit operator/data-policy approval before the first remote call.

## Fail-closed API budget

The registered pilot cannot start an unbounded remote run. Every non-empty
`judge run` first reads `GET /api/v1/key`; missing, nonnumeric, or insufficient
`limit_remaining` aborts before any project text is transmitted. The experiment
also holds an exclusive local lock so two processes cannot reserve the same
budget concurrently.

The frozen safety limits are:

- an explicit `--limit` on every run, with a schema maximum of 25 tasks; the
  v8 controller uses 10 for base/arbitration, 5 for expert, and 1 for smoke;
- at most 5,120 conservatively estimated prompt tokens and 1,024 completion
  tokens per request;
- one automatic retry at most, with all attempts included in preflight;
- at most 275,000 estimated tokens and USD 1.50 per invocation;
- a 2x multiplier on registered token prices;
- at most USD 9.25 observed/projected spend for this pilot;
- at least USD 75.00 projected OpenRouter key credit left after every batch.

The USD 9.25 v8 cap uses the conservative decrease from the key's original
USD 100 allowance, including prior failed/unrecorded attempts, rather than only
successful-response files. With more than USD 94.5 remaining before v8, the
whole lineage remains below the original USD 15.00 ceiling. Increasing the cap
requires explicit operator approval.

All four v8 models support `temperature` under `require_parameters=true`, so
each receives `temperature=0`.
Reasoning effort is likewise model-specific and registered rather than inferred
at runtime. The v8 registration requests `minimal` for the two Gemini models,
`low` for Claude Opus 5, and omits reasoning effort for Claude Haiku 4.5 because
that parameter is unsupported for the model.
`budget/ledger.json` freezes the key's initial remaining credit. Preflight uses
the larger of locally recorded response cost and the decrease in live key
credit, so failed requests or unrelated spending on the same key can only make
the gate more conservative. OpenRouter's provider-reported `usage.cost` is used
when present; registered token prices are the fallback. To change a limit, copy
the registration and prepare a new artifact rather than editing a running
study.

After approval, load the existing secret without printing it:

```bash
set -a
source /scr/del6500/J-lens/runtime/llm-judge/secrets.env
set +a
```

## End-to-end commands

Set a convenience path that does not change the registration:

```bash
JUDGE_CONFIG=Concept_intervention/configs/qwen35_4b_j_raptor_llm_judge_pilot_v8.yaml
jlens-workspace judge validate --config "$JUDGE_CONFIG"
jlens-workspace judge prepare --config "$JUDGE_CONFIG"
```

First run one non-formal response per base judge. `--smoke` requires an
explicit limit and writes only beneath `smoke_responses/`:

```bash
jlens-workspace judge run --config "$JUDGE_CONFIG" --role primary \
  --task-set pointwise --split validation --smoke --limit 1 --jobs 1
jlens-workspace judge run --config "$JUDGE_CONFIG" --role secondary \
  --task-set pointwise --split validation --smoke --limit 1 --jobs 1
```

Run one budgeted validation batch for each base judge and task set:

```bash
for JUDGE_ROLE in primary secondary; do
  jlens-workspace judge run --config "$JUDGE_CONFIG" --role "$JUDGE_ROLE" \
    --task-set pointwise --split validation --limit 10 --jobs 2
  jlens-workspace judge run --config "$JUDGE_CONFIG" --role "$JUDGE_ROLE" \
    --task-set pairwise --split validation --limit 10 --jobs 2
done
```

Rerun the same block until all four response indexes report `complete: true`.
A limited batch exits successfully with `invocation_succeeded: true` even while the
whole task set remains incomplete. Then apply the calibration gate:

```bash
jlens-workspace judge calibrate --config "$JUDGE_CONFIG"
```

Only a passing `calibration.json` permits formal test calls. Run one test batch:

```bash
for JUDGE_ROLE in primary secondary; do
  jlens-workspace judge run --config "$JUDGE_CONFIG" --role "$JUDGE_ROLE" \
    --task-set pointwise --split test --limit 10 --jobs 2
  jlens-workspace judge run --config "$JUDGE_CONFIG" --role "$JUDGE_ROLE" \
    --task-set pairwise --split test --limit 10 --jobs 2
done
```

Rerun this block until all four test indexes report `complete: true`.
The calibration file is checked before every non-smoke test batch.

Freeze arbitration tasks after both base test judges finish, then rerun the
bounded arbitration command until its index reports `complete: true`:

```bash
jlens-workspace judge review-prepare --config "$JUDGE_CONFIG" --tier arbitration
jlens-workspace judge run --config "$JUDGE_CONFIG" --role arbitration \
  --task-set arbitration --limit 10 --jobs 2
```

Only then freeze expert-review tasks. Rerun the smaller expert batch until
`complete: true`:

```bash
jlens-workspace judge review-prepare --config "$JUDGE_CONFIG" --tier expert_review
jlens-workspace judge run --config "$JUDGE_CONFIG" --role expert_review \
  --task-set expert_review --limit 5 --jobs 2
```

Export a deterministic method-blind human subset and aggregate:

```bash
jlens-workspace judge audit-export --config "$JUDGE_CONFIG"
jlens-workspace judge aggregate --config "$JUDGE_CONFIG"
```

The base pilot requires 896 validation calls and 896 test calls per base judge,
or 3,584 base calls in total. Arbitration/expert calls are data-dependent.
Every result index records provider-reported tokens, provider-reported or
registered-price cost, the conservative preflight bound, and remaining task
count; routing never depends on price.

## Artifact contract

`llm_judge_pilot_v8/` contains:

```text
manifest.json
migration/v7_verified_import.json     # only after exact hash verification
tasks/
  pointwise_{validation,test}.jsonl
  pairwise_{validation,test}.jsonl
  arbitration.jsonl                 # after review preparation
  expert_review.jsonl               # after expert preparation
private/
  *_map.jsonl
responses/<exact-model>/<task-set>/*.json
smoke_responses/<exact-model>/<task-set>/*.json
budget/run.lock
budget/ledger.json
calibration.json
human_audit/{tasks,annotations_template}.jsonl
human_audit/manifest.json
analysis/{pointwise_unblinded,pairwise_unblinded}.jsonl
analysis/{report,index}.json
```

Preparation validates candidate-score and generation hashes, row cardinality,
telemetry coverage, selected layers, prompt/decoding identity, and token-level
equivalence of J zero and RAPTOR no-hook outputs. The manifest seals every
selected source summary/generation file, public/private task file, rubric,
registration, and judge implementation source file. Re-running preparation is
idempotent only when every frozen row is identical.

Every response records `validation_warnings`. A nonempty warning may only
describe evidence items removed by the registered evidence sanitizer or excess
rationale characters truncated at the registered 600-character limit. All
estimator fields remain fail-closed. Warning counts are retained for audit and
must be reported with the final analysis.

Imported v7 responses retain their original usage, provider, attempts, warnings,
task hash, and rubric hash, and count against the v8 experiment budget. The
migration manifest seals the source/destination task-file hashes and every copied
response before the v8 controller starts. No response is imported when any
scientific identity field differs.

Do not report the lane as complete until `analysis/index.json` exists with
`complete: true`. Prepared tasks, a passing API smoke, or a passing calibration
alone are not experimental findings.
