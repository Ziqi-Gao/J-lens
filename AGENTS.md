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

The root direction directories are scientific control surfaces, not Python
source mirrors. They retain experiment identity, design YAML, small input
manifests, protocols, launch transports, and reports. Reusable implementation
belongs under `src/jlens_workspace/`: common model/Jacobian/artifact services in
`foundation/`, shared dataset interfaces in top-level `data/`, Concept
Intervention code in the explicit `data/`, `protocol/`, `probing/`, `geometry/`,
`steering/`, `evaluation/`, and `reporting/` responsibilities, and J-space
operator/spectrum/subspace code in `j_space/`. Historical import paths may
remain only as tested compatibility wrappers; new logic uses the canonical
paths. See `docs/architecture.md`.

## Server ownership and project isolation

J-lens is an independent project with exactly three writable directory trees:

1. `/home/del6500/projects/J-lens` for Git-controlled code and small metadata;
2. `/data/del6500/J-lens` for persistent datasets and durable artifacts; and
3. `/scr/del6500/J-lens` for environments, caches, runtime state, temporary
   checkpoints, and logs.

The main checkout exposes two server-local convenience links: `data` must
resolve exactly to `/data/del6500/J-lens`, and `scratch` must resolve exactly
to `/scr/del6500/J-lens`. They are ignored by Git and are entry points only;
the absolute trees above remain the authoritative ownership boundary. Project
Codex defaults belong in `.codex/config.toml`, while durable repository policy
belongs in this file.

Start every J-lens Codex task with the runtime workspace set exactly to
`/home/del6500/projects/J-lens`. Do not resume or reuse a task rooted at
`/home`, `/home/del6500/projects`, or another project. The project-local
`jlens-isolated` permission profile gives the host a read-only baseline and
reopens write access only for the three J-lens trees above. It redirects
temporary files to `/scr/del6500/J-lens/tmp`, disables command network access,
and keeps `.codex` and `.git` read-only.

"Approve for me" is allowed with this profile: `sandbox_approval` and
`request_permissions` are auto-rejected before review. Do not switch profiles,
pass `--sandbox`, use a danger-full-access or sandbox-bypass option, or weaken
those two gates. A protected configuration or Git-metadata change requires a
separate, explicitly user-approved maintenance operation and must not broaden
the three owned roots.

Create linked Git worktrees only below `/scr/del6500/J-lens/worktrees/`. Do not
create sibling checkout trees below `/home/del6500/projects`, and do not place
worktree contents inside the main repository's `.git` directory. Before
removing a worktree, require a clean checkout, a preserved branch ref, no live
process or service path reference, and durable artifacts or an immutable
runtime snapshot where applicable. Remove it with `git worktree remove`, never
by recursively deleting the directory.

Everything outside those trees is read-only and must never be written by a
J-lens agent. This includes `/home/del6500/projects/FedFisher`,
`/data/del6500/FedFisher`, and `/scr/del6500/FedFisher`. J-lens and FedFisher
must not share environments, caches, locks, logs, data, artifacts, launchers,
or maintenance policy.

A single agent, task, or session that writes J-lens must not also modify
FedFisher. If a request requires changes in both projects, stop and divide the
work into separately owned agents/tasks before either side is edited. Read-only
inspection of another project's maintenance rules is allowed when necessary;
it does not authorize a write. Before writing through a symlink, resolve its
target and require that it remains inside one of the three J-lens trees.

Keep large or durable outputs out of the code checkout. Pretrained download
caches and build/runtime files belong under `/scr/del6500/J-lens`; prepared
scientific data, fitted lenses, generation shards, and final indexes belong
under `/data/del6500/J-lens`. Do not install packages globally or alter user
shell configuration.

## Shared GPU non-interference

Processes, Screen/tmux sessions, containers, and GPU contexts not launched by
the current J-lens task are foreign experiments. Existing GPU use does not by
itself prohibit a J-lens overlay: the user has authorized additive execution
even when it lengthens a foreign training step, provided memory headroom and
resource admission make task failure unlikely. Never encode a status-derived
GPU number in a launcher. Discover visible devices dynamically; treat
`JLENS_LOCAL_GPU_IDS` only as an optional operator allow-list.

Use the checked-in local scheduler, J-lens-owned locks and leases, and an
explicit `CUDA_VISIBLE_DEVICES`. It must classify J-lens versus foreign compute
PIDs, record the pre-existing snapshot, take rolling min-free/max-utilization
samples, rank clean devices before bounded overlays, acquire CPU/RAM/GPU
tokens, and revalidate after taking the device lock. Every new shard repeats
admission, so a newly freed GPU joins automatically. Task profiles distinguish
CPU-heavy, low-memory occupancy from exclusive model/capture/generation work.

Never terminate, signal, pause, renice, debug, or change the affinity,
environment, files, clocks, power settings, MIG layout, MPS state, or driver
state of a foreign process. Never use GPU reset, `killall`, or broad `pkill`.
The scheduler does not preempt a task in the middle of a scientific shard. If
a J-lens overlay causes OOM or instability, adjust or stop only the exact
J-lens process/session launched by the current task after verifying ownership,
then increase the affected task profile or reserve before retrying. Leave every
foreign workload untouched.

Within J-lens, ordinary GPU tasks remain mutually exclusive per device.
Measured low-memory, CPU-heavy occupancy tasks may use a bounded shared gate
with explicit per-device slots and utilization/memory tokens. CPU and host RAM
leases apply to GPU and CPU tasks alike. Profile, token, worker-pool, or
slot-count changes require a clean commit and a controlled DAG restart; never
hot-edit a running checkout.

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

## Experiment identity and versioning

The canonical inventory is
`Concept_intervention/experiments/registry.yaml`; its policy and layout are in
`Concept_intervention/experiments/README.md`. Before creating or changing a
versioned experiment, run `jlens-workspace experiments list` and inspect the
registered family.

Keep four identities separate:

- `design-vN` is only for a scientific change to the hypothesis, method, data
  or split, selection rule, estimand, primary metric, or judge design;
- `protocol-vN` is for a compatible prompt, rubric, schema, or transport
  revision;
- `attempt-NNN` is for retrying or resuming the same frozen design after an
  interruption, timeout, OOM, API failure, or parser repair; and
- `revision-rN` is for a derived rescore or reanalysis over immutable source
  artifacts.

A retry must never become a new `design-vN`. Git SHAs, dates, snapshot hashes,
and server names are provenance fields, not experiment versions. New designs
belong under
`Concept_intervention/experiments/<family>/<design-version>/`; do not add new
versioned files to the legacy flat `configs/` and `scripts/` surfaces.
Existing registered paths and artifacts remain immutable compatibility inputs.

Every new design must add its `experiment.yaml` to the registry in the same
commit and pass `jlens-workspace experiments validate`. The managed server
audit additionally runs `jlens-workspace experiments validate
--check-artifacts`; this is read-only and does not turn runtime state into a
completion marker.

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
