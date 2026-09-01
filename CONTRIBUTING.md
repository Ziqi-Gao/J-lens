# Contributing

Contributions should make the two research programs more reproducible without
blurring their experimental boundaries.

## Set up

Use Python 3.11+ and `uv` from the repository root:

```bash
uv sync --extra dev
.venv/bin/pytest
.venv/bin/ruff check .
```

For model, remote-data, or GPU work:

```bash
uv sync --extra dev --extra llm
.venv/bin/jlens-workspace doctor
```

Ordinary tests must remain fast and offline. Mark tests that require a model
download, remote data, or CUDA with the existing `integration`, `remote`, or
`gpu` marker.

## Respect the repository boundary

- Keep `Concept_intervention/` and `J_space/` as root-level siblings.
- Put reusable typed logic in `src/jlens_workspace/` and synthetic tests in
  `tests/`.
- Treat the root direction directories as scientific control surfaces: they
  retain experiment identity, configs, small manifests, protocols, launchers,
  and reports even when their reusable computation lives under `src/`.
- Put new Concept Intervention code in the appropriate `probing`, `geometry`,
  `steering`, `evaluation`, or `reporting` namespace. Do not add a generic
  `methods` package that conflates probing with steering.
- Put only direction-specific data manifests, YAML, launchers, and reports in
  the corresponding direction directory.
- Do not import one direction's scripts or generated outputs from the other.
- Do not put reusable experimental logic in notebooks or Slurm files; launchers
  should validate configuration and call `jlens-workspace`.
- Preserve historical import paths only as thin compatibility wrappers; new
  implementation code must use the canonical paths documented in
  [`docs/architecture.md`](docs/architecture.md).
- On the managed local server, obey the three J-lens write roots and project
  isolation policy in `AGENTS.md`. An agent writing J-lens must not write
  FedFisher in the same task or session.

## Non-negotiable scientific invariants

1. Capture and intervene at transformer block output, recorded as
   `resid_post`, for J-lens claims. A faithful external baseline may retain a
   different published coordinate only when it is explicitly named, isolated,
   attributed, and documented as a method-specific comparison.
2. Keep the official Jacobian-lens dependency pinned unless a migration is
   intentional, tested, and documented.
3. Reject a saved lens if model/tokenizer revisions, residual coordinate, BOS
   policy, target layer, or weight coordinates differ.
4. Never materialize a production \(V\times D\) token-frame matrix. Iterate
   vocabulary chunks and accumulate a \(D\times D\) Gram matrix.
5. Accumulate spectrum statistics in float64 by default and record the lens
   storage precision.
6. Preserve semantic groups across splits. Select probe hyperparameters on
   train only and evaluate test once after the choice is fixed.
7. Export standardized probe coefficients back into original residual
   coordinates.
8. Include random/non-J matched controls before making a causal steering claim.
9. Record the exact matrix convention; raw, centered, row-normalized, and
   RMSNorm-weighted objects are not interchangeable.
10. Write a provenance manifest for every expensive run.

## Data changes

Every concept record must satisfy the schema in
[`Concept_intervention/docs/data.md`](Concept_intervention/docs/data.md). For
paired examples, both labels in a group must use the same label-neutral prompt
and split. Avoid lexical or topic shortcuts: a negative should answer the same
situation without instantiating the target concept, not switch to an unrelated
subject.

Do not enable an AxBench entry from a guessed word or ID. Discover candidates
at the pinned revision, inspect positive examples, then record the exact remote
`output_concept` and verification revision. Preserve upstream source and
license metadata.

## Experiment identity

All versioned Concept Intervention work must be registered in
[`Concept_intervention/experiments/registry.yaml`](Concept_intervention/experiments/registry.yaml).
Use `design-vN` only for scientific design changes. Use `protocol-vN` for a
compatible schema/rubric change, `attempt-NNN` for retries and resumes, and
`revision-rN` for derived rescoring or reanalysis.

New designs live under
`Concept_intervention/experiments/<family>/<design-version>/` and add their
manifest to the registry in the same commit. Do not extend the historical flat
`configs/` and `scripts/` version pattern. Existing paths remain as immutable
compatibility surfaces. Validate metadata with:

```bash
.venv/bin/jlens-workspace experiments validate
```

## Artifact or config changes

- Schema/config changes must reject unknown keys and include migration notes.
- A new matrix variant needs a distinct name, output directory, and explicit
  coordinate definition.
- Existing artifacts are immutable. Do not add an overwrite default or reuse a
  scientific run directory.
- Never commit model weights, lenses, activation arrays, parquet files, caches,
  Slurm logs, or generated artifacts.
- New cluster launchers must accept config/cache overrides and pass `bash -n`.

## Pull-request checklist

Before requesting review:

```bash
.venv/bin/jlens-workspace experiments validate
.venv/bin/ruff check .
.venv/bin/pytest
bash -n Concept_intervention/scripts/*.sh
bash -n Concept_intervention/scripts/*.slurm
bash -n J_space/scripts/*.sh
bash -n J_space/scripts/*.slurm
```

In the change description, state:

- the research question and whether this is reproduction or extension;
- tensor shapes and coordinate conventions;
- model/tokenizer/lens/data revisions and seed;
- selection data versus held-out evaluation data;
- expected artifact paths and storage cost;
- tests and controls run; and
- limitations or failed checks.

Contributions are accepted under the repository's
[Apache-2.0 license](LICENSE).
