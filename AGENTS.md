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
