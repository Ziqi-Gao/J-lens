"""Compatible facade for the fail-closed protocol-v2 ServerScheduler adapter.

The implementation is split by responsibility under :mod:`jlens_workspace.scheduler`.
This module deliberately preserves the original import and ``python -m`` surface used
by the checked-in entrypoint.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence

from jlens_workspace.foundation.artifacts import sha256_file
from jlens_workspace.scheduler.allocation import Allocation
from jlens_workspace.scheduler.catalog import (
    CODE_ROOT,
    COMPLETION_ROOT,
    CPU_BOOTSTRAP,
    CPU_DEFAULT,
    CPU_HEAVY,
    DATA_ROOT,
    GPU_KDIAG_BUNDLE,
    GPU_MODEL,
    GPU_OCCUPANCY,
    GPU_STANDARD,
    GPU_STANDARD_SHARD,
    KDIAG_RUN_ID,
    KDIAG_RUN_ROOT,
    PROJECT,
    PROJECT_PYTHON,
    SCRATCH_ROOT,
    TASK_SPECS,
    THREE_METHOD_RUN_ID,
    THREE_METHOD_RUN_ROOT,
    ResourceProfile,
    TaskSpec,
)
from jlens_workspace.scheduler.completion import (
    CompletionEvidence,
    validate_task_completion,
)
from jlens_workspace.scheduler.errors import ServerSchedulerAdapterError
from jlens_workspace.scheduler.manifest import (
    RunningJob,
    load_running_job,
)
from jlens_workspace.scheduler.runner import (
    DispatchPlan,
    build_dispatch_plan,
    execute_dispatch_plan,
)
from jlens_workspace.scheduler.runner import (
    require_clean_checkout as _require_clean_checkout,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate and run one scheduler-provided fixed task body."""

    arguments = _parser().parse_args(argv)
    try:
        plan = build_dispatch_plan(arguments.job_manifest)
        return execute_dispatch_plan(plan)
    except (OSError, ServerSchedulerAdapterError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


__all__ = [
    "CODE_ROOT",
    "COMPLETION_ROOT",
    "CPU_BOOTSTRAP",
    "CPU_DEFAULT",
    "CPU_HEAVY",
    "DATA_ROOT",
    "GPU_KDIAG_BUNDLE",
    "GPU_MODEL",
    "GPU_OCCUPANCY",
    "GPU_STANDARD",
    "GPU_STANDARD_SHARD",
    "KDIAG_RUN_ID",
    "KDIAG_RUN_ROOT",
    "PROJECT",
    "PROJECT_PYTHON",
    "SCRATCH_ROOT",
    "TASK_SPECS",
    "THREE_METHOD_RUN_ID",
    "THREE_METHOD_RUN_ROOT",
    "Allocation",
    "CompletionEvidence",
    "DispatchPlan",
    "ResourceProfile",
    "RunningJob",
    "ServerSchedulerAdapterError",
    "TaskSpec",
    "_require_clean_checkout",
    "build_dispatch_plan",
    "execute_dispatch_plan",
    "load_running_job",
    "main",
    "sha256_file",
    "subprocess",
    "validate_task_completion",
]


if __name__ == "__main__":
    raise SystemExit(main())
