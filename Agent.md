# Agent operating guide

This file is the compact task-entry guide for coding and research agents. Read
[`README.md`](README.md), [`docs/design.md`](docs/design.md), and the relevant
direction README before acting.

## Map the task first

- Concept data, probes, alignment, or intervention:
  [`Concept_intervention/`](Concept_intervention/README.md).
- Matrix rank, singular spectrum, PCA/energy bases, or layerwise subspaces:
  [`J_space/`](J_space/README.md).
- Logic shared by both: `src/jlens_workspace/` with tests in `tests/`.
- Shared research decisions: `docs/`.

The two direction directories are root-level siblings. Never create
`Concept_intervention/J_space/` or make either direction depend on the other's
scripts/artifacts.

## Before editing

1. Inspect the current tree and dirty changes; preserve concurrent user/agent
   work.
2. Read the configuration model and the artifact producer/consumer involved.
3. State the tensor coordinate and shape at every boundary.
4. Decide which check can run offline before attempting a GPU or remote run.
5. Keep optional Torch/Transformers/Datasets/J-lens imports lazy.

## Hard guards

- Official J-lens pin:
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- Residual coordinate: transformer block output / `resid_post`.
- Lens reuse requires exact model, tokenizer, layer, target, BOS, normalization,
  and weight-coordinate compatibility.
- Production \(V\times D\) matrices are streamed, never materialized.
- Spectrum Gram accumulation defaults to float64.
- Rectangular \(A_l\) has singular values, not eigenvalues; eigenanalysis is of
  \(A_l^\top A_l\).
- Probe selection uses train only; groups never cross splits; test is touched
  once after selection.
- Scientific interventions need signed strengths and matched full/J/non-J/random
  controls.
- Results without complete provenance are incomplete.

## Preferred execution path

Use typed library functions under tests, then a thin CLI/config surface:

```bash
.venv/bin/jlens-workspace doctor
.venv/bin/jlens-workspace config validate CONFIG.yaml
.venv/bin/jlens-workspace data validate DATA.jsonl
.venv/bin/jlens-workspace concept run CONFIG.yaml
.venv/bin/jlens-workspace matrix run CONFIG.yaml
```

Use the direction launchers for smoke and cluster runs. Do not silently
overwrite an output or modify a pinned YAML for an ad-hoc run; copy it to a new
config and output directory.

## Definition of done

- Focused offline tests pass; the full ordinary suite has no unexplained
  failure.
- Ruff and shell syntax checks pass for touched files.
- Data changes preserve schema/counts/group semantics and avoid shortcuts.
- Documentation states assumptions, provenance, run command, and artifact path.
- GPU/remote work is either verified or explicitly reported as not run.
- The handoff distinguishes evidence observed locally from expected behavior.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full review checklist.
