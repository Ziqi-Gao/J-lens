# J-lens Workspace

Reproducible experiments for two related questions raised by
[Jacobian Lens](https://github.com/anthropics/jacobian-lens) and the
[global-workspace study](https://transformer-circuits.pub/2026/workspace/):

1. Can a J-lens help identify and causally steer abstract semantic concepts?
2. What is the spectral and subspace geometry of the token-facing J-space
   matrix?

This is a research workspace, not a claim that a linear readout is a complete
description of model cognition. Correlational probe/alignment results and
causal intervention results are kept distinct.

## Prior work and this repository's scope

Neither direction is presented as unprecedented. The 2026
[Workspace study](https://transformer-circuits.pub/2026/workspace/index.html)
already:

- extracted concept vectors and inferred-intermediate probes, decomposed them
  as sparse non-negative combinations of J-lens vectors plus non-J residuals,
  and intervened on both components (Figures 8 and 16); the J component carried
  only about 6–7% of concept-vector variance and roughly 10–15% of
  inferred-intermediate probe variance, yet concentrated much more of the
  causal effect; and
- measured, in Figure 28d, the fraction of residual dimensions needed to
  capture variance across \(W_UJ_l\), an effective linear-dimensionality
  analysis closely related to this repository's second direction.

The earlier
[Emergent Introspective Awareness](https://arxiv.org/abs/2601.01828) study also
used concept-vector injection, including abstract nouns such as justice, peace,
betrayal, balance, and tradition.

This repository is an open-model reproduction and extension on pinned Qwen
checkpoints. Its intended additions are reproducible group-safe logistic
probes, train-only selection and held-out tests, explicit rank/effective-rank
and singular-basis reports, matched controls, and tests of whether findings
generalize across abstract concepts and layers. A result should be described as
a reproduction, failure to reproduce, or scoped extension—not as discovery of
concept/J-space decomposition or effective dimensionality itself.

## Two sibling research programs

| Direction | Question | Entry point | Default artifacts |
| --- | --- | --- | --- |
| Abstract-concept intervention | Does a residual concept direction align with a sparse combination of token J-directions, and does intervening on it change semantics? | [`Concept_intervention/`](Concept_intervention/README.md) | `artifacts/concept_intervention/` |
| J-space geometry | How do rank, singular spectrum, effective dimension, PCA bases, and layer-to-layer subspaces behave? | [`J_space/`](J_space/README.md) | `artifacts/j_space/` |

`Concept_intervention/` and `J_space/` must remain root-level siblings. Shared,
tested code lives in `src/jlens_workspace/`; neither direction imports scripts
or generated outputs from the other.

## Mathematical contract

At transformer layer \(l\), let

- \(J_l \in \mathbb{R}^{D \times D}\) be the fitted average Jacobian that maps
  the layer-\(l\) `resid_post` coordinate into the target residual frame;
- \(U_{\mathrm{eff}} \in \mathbb{R}^{V \times D}\) be the explicitly recorded
  effective unembedding restricted to tokenizer IDs `[0,V)`; padded model-head
  rows with no tokenizer ID are excluded; and
- \(A_l = U_{\mathrm{eff}}J_l \in \mathbb{R}^{V \times D}\) be the token-facing
  J-space matrix.

The supported coordinate conventions are:

- `raw`: \(U_{\mathrm{eff}} = W_U\);
- `rmsnorm_weighted`: \(U_{\mathrm{eff}} = W_U\operatorname{diag}(\gamma)\).

For `rmsnorm_weighted`, the input-dependent RMS scalar is not folded into the
matrix. A lens is compatible only with its recorded model and tokenizer
revisions, layer/target layer, `resid_post` hook location, BOS policy, and model
weight coordinates.

Because \(A_l\) is generally rectangular, this project does not report its
eigenvalues. It streams vocabulary blocks, accumulates the \(D \times D\) Gram
matrix \(G_l=A_l^\top A_l\) in float64, and reports eigenvalues of \(G_l\), or
equivalently singular values \(\sigma_i(A_l)=\sqrt{\lambda_i(G_l)}\). Centered,
row-normalized, and unnormalized matrices are separately named objects.

For concept steering, each labeled text is encoded by the pinned model
tokenizer and the final non-padding token at `resid_post` is used directly;
there is no sequence or response pooling. A leakage-safe probe yields a
direction \(w_l\in\mathbb{R}^D\) in original `resid_post` coordinates. The
current formal workflow compares \(w_l\) with every streamed row of \(A_l\) and
reports cosine-ranked tokens. Sparse decomposition and causal intervention are
later, separate stages.

See [research design](docs/design.md) and the
[experiment protocol](docs/experiment_protocol.md) for the full invariants and
evidence gates.

## Repository layout

```text
Concept_intervention/   data, configs, launchers, and reports for steering
J_space/                configs, launchers, and reports for matrix geometry
src/jlens_workspace/    shared typed implementation
tests/                  fast offline tests plus opt-in LLM/GPU tests
scripts/                tiny public-model integration checks
docs/                   shared design and experiment protocol
artifacts/               generated immutable run outputs; gitignored
```

## Quick start

Requirements are Python 3.11+ and
[`uv`](https://docs.astral.sh/uv/). Core tests do not import Torch,
Transformers, Datasets, or the external J-lens package.

```bash
uv sync --extra dev
.venv/bin/jlens-workspace doctor
.venv/bin/pytest
.venv/bin/ruff check .
```

Install the model/GPU stack only when needed:

```bash
uv sync --extra dev --extra llm
```

The `llm` extra pins the official Jacobian-lens repository to commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`.

### Truly offline smoke

These checks force Hugging Face offline mode. They validate configs, the
human-written 96-row concept dataset, split/group invariants, and synthetic
workflow tests without downloading a checkpoint:

```bash
Concept_intervention/scripts/smoke_offline.sh
J_space/scripts/smoke_offline.sh
```

The J-space numerical tests run when Torch is installed and are reported as
skipped otherwise. Config/data validation remains active in the core-only
environment.

### Tiny real-model integration

The tiny scripts fit an actual J-lens on the pinned
`sshleifer/tiny-gpt2` checkpoint. The first run needs network access; later runs
can be offline if the checkpoint is already in `HF_HOME`.

```bash
J_space/scripts/run_tiny_integration.sh
Concept_intervention/scripts/run_tiny_integration.sh
```

To prove that the cache is sufficient:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  J_space/scripts/run_tiny_integration.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  Concept_intervention/scripts/run_tiny_integration.sh
```

These are dependency/API integration checks, not scientific experiments.

## Qwen3.5-4B on the cluster

The checked-in Qwen configs pin the model and tokenizer revisions. The formal
concept config refits its own float32 J-lens from 1,000 label-blind prompts,
which may overlap probe texts, and restricts the same model's unembedding to
actual tokenizer IDs. Before
submitting, install the `llm` extra on the login node and choose a shared
Hugging Face cache with enough space:

```bash
uv sync --extra dev --extra llm
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
```

Quest launchers request account `p32737`, partition `gengpu`, one A100, 8 CPUs,
and 64 GB RAM. The concept job is checkpointed because local J-lens fitting is
the dominant cost:

```bash
sbatch Concept_intervention/scripts/run_qwen35_4b.slurm
sbatch J_space/scripts/run_qwen35_4b.slurm
```

The same commands can run inside an existing interactive GPU allocation:

```bash
Concept_intervention/scripts/run_qwen35_4b.sh
J_space/scripts/run_qwen35_4b.sh
```

Override a config without editing a launcher:

```bash
CONCEPT_CONFIG=/absolute/path/to/concept.yaml \
  sbatch Concept_intervention/scripts/run_qwen35_4b.slurm
JSPACE_CONFIG=/absolute/path/to/matrix.yaml \
  sbatch J_space/scripts/run_qwen35_4b.slurm
```

Each launcher runs `doctor --require-llm`, a CUDA bfloat16 preflight, and strict
config validation before model work; the concept launcher also validates the
377,027-row audit artifact / 53,861-source full GoEmotions dataset and its
50,000-example-per-concept lower bound, then fits or resumes the local lens.
The J-space launcher defaults to the
centered PCA configuration. A scientific rerun should use a copied
config with a new `experiment_name` and `output_dir`, rather than overwriting a
prior artifact. If compute nodes lack outbound access, pre-populate `HF_HOME`
with every pinned model and lens file before submission.

## CLI surface

The direction launchers call the CLI rather than duplicating workflow logic:

```bash
jlens-workspace doctor
jlens-workspace config validate CONFIG.yaml
jlens-workspace data validate DATA.jsonl
jlens-workspace lens fit CONFIG.yaml
jlens-workspace concept capture CONFIG.yaml
jlens-workspace concept fit-probes CONFIG.yaml
jlens-workspace concept align CONFIG.yaml
jlens-workspace concept run CONFIG.yaml
jlens-workspace matrix run CONFIG.yaml
```

Use the staged concept commands to inspect intermediate artifacts. The current
`concept run` launcher executes capture, probe fitting, and J/non-J alignment;
generation-time intervention is available as a library primitive and is not
silently implied by that command.

## Formal and smoke concept data

The formal dataset uses seven full one-vs-rest GoEmotions tasks over 53,861
unique split-safe texts, including multilabel examples. Natural positive counts
range from 1,967 to 5,093; the former 2,000-row balanced tasks are controls.
The built-in dataset contains
honesty, justice, compassion, uncertainty,
creativity, and power, with positive and negative examples in every split. It
is an offline smoke/research seed, not enough evidence for a general steering
claim. Its exact schema and preparation API are documented in the
[data guide](Concept_intervention/docs/data.md).

The checked-in AxBench allowlist intentionally has
`remote_output_concept=null` and `verified_against_revision=null`. Local names
such as “honesty” have not yet been manually matched to exact positive
GemmaScope feature descriptions at the pinned Concept16K revision. Therefore
`prepare_axbench` fails before importing `datasets` or touching the network.
This prevents fabricated remote IDs and accidental labeling of a concrete or
polysemous feature as an abstract concept. Use read-only discovery, inspect
examples, then record the exact remote description and verified revision.

## Artifact contract

Generated outputs are written below the config's `output_dir` and are ignored
by Git. The logical layout is:

```text
artifacts/
├── concept_intervention/<run>/
│   ├── manifest.json      immutable pins, config hash, environment provenance
│   ├── run.json           completed stage index
│   ├── activations/       layer_XX.npy, labels.npy, rows.jsonl, metadata
│   ├── probes/            raw-coordinate probe vectors and held-out metrics
│   ├── alignment/         token rankings and J/non-J decompositions
│   └── interventions/     generations, controls, and dose-response results
└── j_space/<run>/
    ├── manifest.json
    ├── metrics.json
    └── layer_NNN/
        ├── metrics.json
        ├── singular_values.npy
        ├── eigenvalues.npy
        ├── basis_numerical_rank.npy
        ├── center_mean.npy        # only when centered
        └── basis_energy_*.npy
```

Every expensive run must retain provenance: exact model/tokenizer/lens/data
revisions, dataset fingerprint, layers, coordinate convention, seed, dtypes,
package versions, and Git commit when available. Large tensors and caches must
never be committed.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing an invariant or artifact
format. Automation and coding agents should also read [AGENTS.md](AGENTS.md).
This repository is licensed under [Apache License 2.0](LICENSE).
