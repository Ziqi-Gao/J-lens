# Agent guide

## Purpose

This repository supports two sibling research programs:

- `Concept_intervention/`: abstract-concept probes, J-lens alignment, and causal
  steering.
- `J_space/`: spectral and subspace analysis of `A_l = U_eff J_l`.

Shared, tested code belongs in `src/jlens_workspace/`. Direction-specific YAML,
launchers, small data manifests, and reports belong in the corresponding root
directory. Do not recreate `Concept_intervention/J_space/`.

Before acting, read [`README.md`](README.md), [`docs/design.md`](docs/design.md),
and the README for the relevant direction. Route concept data, probes,
alignment, and intervention work to `Concept_intervention/`; route rank,
singular-spectrum, PCA/energy-basis, and layerwise-subspace work to `J_space/`.
Neither direction may depend on the other direction's scripts or artifacts.

## Server ownership and project isolation

J-lens is an independent project with exactly three writable directory trees:

1. `/home/del6500/projects/J-lens` for Git-controlled code and small metadata;
2. `/data/del6500/J-lens` for persistent datasets and durable artifacts; and
3. `/scr/del6500/J-lens` for environments, caches, runtime state, temporary
   checkpoints, and logs.

Everything outside those trees is read-only for a J-lens agent. This includes
`/home/del6500/projects/FedFisher`, `/data/del6500/FedFisher`, and
`/scr/del6500/FedFisher`. J-lens and FedFisher must not share environments,
caches, locks, logs, data, artifacts, launchers, or maintenance policy.

A single agent, task, or session that writes J-lens must not also modify
FedFisher. If a request requires changes in both projects, stop and divide the
work into separately owned agents/tasks before either side is edited. Read-only
inspection of another project's public maintenance rules is allowed when it is
necessary to preserve server policy; it does not authorize a write there.
Before writing through a symlink, resolve its target and require that it remains
inside one of the three J-lens trees.

Keep large or durable outputs out of the code checkout. Pretrained download
caches and build/runtime files belong under `/scr/del6500/J-lens`; prepared
scientific data, fitted lenses, generation shards, and final indexes belong
under `/data/del6500/J-lens`. Do not install packages globally or alter user
shell configuration.

## Shared GPU non-interference

Processes, Screen/tmux sessions, containers, and GPU contexts not launched by
the current J-lens task are foreign experiments. Existing GPU use does not by
itself prohibit a J-lens overlay: the user has authorized additive execution
when stable free-memory and utilization thresholds leave sufficient capacity.
Use the checked-in local GPU wrapper, a J-lens-owned lock, and an explicit
`CUDA_VISIBLE_DEVICES`; record the pre-existing compute-process snapshot.

Never terminate, signal, pause, renice, debug, or change the affinity,
environment, files, clocks, power settings, MIG layout, MPS state, or driver
state of a foreign process. Never use GPU reset, `killall`, or broad `pkill`.
If a J-lens overlay causes OOM, instability, or material contention, adjust or
stop only the exact J-lens process/session launched by the current task after
verifying its ownership. Leave every foreign workload untouched.

Within J-lens, ordinary GPU tasks remain mutually exclusive per device.
Measured low-memory, CPU-heavy occupancy tasks may use a bounded shared gate
with explicit per-device slots, while retaining the same capacity sampling and
foreign-process non-interference checks. Slot-count changes require a clean
commit and a controlled DAG restart; never hot-edit a running checkout.

## Non-negotiable invariants

1. Pin the official Jacobian-lens implementation to commit
   `581d398613e5602a5af361e1c34d3a92ea82ba8e` unless an intentional migration is
   documented.
2. Capture and intervene at block output / `resid_post`.
3. Never use a prefitted lens with a different model revision, tokenizer,
   coordinate-changing model wrapper, or BOS policy.
4. Never materialize a production `V x D` matrix. Iterate vocabulary chunks or
   accumulate a `D x D` Gram matrix.
5. Accumulate spectral statistics in float64 by default. Do not infer a tail
   rank from a lens saved only in float16.
6. Tune logistic-probe `C` on training data only. Preserve group boundaries and
   map standardized coefficients back to original residual coordinates.
7. Keep imports of Torch, Transformers, Datasets, and the external `jlens`
   package lazy so core data and numerical tests run without a GPU stack.
8. Every result must include a provenance manifest and the matrix coordinate
   convention.
9. `A_l` is rectangular: analyze its singular values or the eigenvalues of
   `A_l.T @ A_l`, never purported eigenvalues of `A_l` itself.
10. A causal intervention claim requires signed strengths and matched full,
    J, non-J, and random controls. Probe AUC or nearest-token alignment alone
    is not causal evidence.

## Development workflow

Use Python 3.11+ and `uv`:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Add `--extra llm` for activation, J-lens, and intervention runs. Tests marked
`integration`, `remote`, or `gpu` are opt-in; ordinary unit tests must be fast
and offline.

New features should expose a small typed function in `src/jlens_workspace/`, add
synthetic unit tests, then add a thin YAML-driven experiment entrypoint. Avoid
putting reusable logic in notebooks or Slurm scripts.

Before editing, inspect the worktree and preserve unrelated changes. Read both
the artifact producer and consumer, state tensor shapes and coordinates at each
boundary, and choose an offline check before a GPU or remote run. Do not
silently overwrite an output or edit a pinned scientific YAML for an ad-hoc
run; copy it to a new config and output directory.

The preferred command surface is:

```bash
.venv/bin/jlens-workspace doctor
.venv/bin/jlens-workspace config validate CONFIG.yaml
.venv/bin/jlens-workspace data validate DATA.jsonl
.venv/bin/jlens-workspace concept run CONFIG.yaml
.venv/bin/jlens-workspace matrix run CONFIG.yaml
```

## Review checklist

- Are tensor shapes and coordinate systems stated in docstrings?
- Are model, tokenizer, lens, data revision, layer, seed, and dtype recorded?
- Is tuning isolated from held-out evaluation?
- Does large-vocabulary work use chunking?
- Is there a CPU path for numerical tests and a clear GPU path for real runs?
- Are random and non-J controls present before a causal claim is made?

A change is done only when focused offline tests and the ordinary suite pass,
Ruff and touched shell syntax checks pass, data schemas/counts/group semantics
remain valid, and documentation records assumptions, commands, provenance, and
artifact paths. Report GPU or remote checks as either observed or not run; do
not present expected behavior as measured evidence. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the full checklist.
