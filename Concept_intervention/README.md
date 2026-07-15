# Abstract-concept intervention

This root-level direction asks whether J-lens geometry can help isolate and
causally steer abstract semantic concepts. It is a sibling of
[`J_space/`](../J_space/README.md), not its parent.

This direction reproduces and extends existing experiments. The
[Workspace study](https://transformer-circuits.pub/2026/workspace/index.html)
already decomposed concept vectors and inferred-intermediate probes into sparse
non-negative J and non-J components, then intervened on both (Figures 8 and
16). Its J components explained only about 6–7% and 10–15% of variance in the
two settings while carrying much more causal effect. The
[introspective-awareness study](https://arxiv.org/abs/2601.01828) previously
injected concept vectors including abstract nouns. Here the goal is a rigorous
open-Qwen replication/extension with paired data, group-safe logistic probes,
held-out evaluation, matched controls, and abstract-concept generalization.

## Research sequence

For each concept and selected layer:

1. validate binary examples, freeze the dataset fingerprint, and preserve
   semantic groups across train/validation/test;
2. tokenize the labeled text with the pinned Qwen tokenizer and capture the
   final non-padding token at transformer block output (`resid_post`); no
   mean/max/span pooling is implemented;
3. tune an L2 logistic probe's inverse penalty `C` with grouped CV on train;
4. inspect validation, refit the fixed `C` on train+validation, and evaluate
   test exactly once;
5. compare the raw-coordinate probe vector with vocabulary rows of
   \(A_l=U_{\mathrm{eff}}J_l\) using chunked search;
6. inspect whether the nearest vocabulary directions are semantically coherent
   with the concept before designing any intervention experiment.

A probe AUC or nearest token is not causal evidence. A steering claim requires
a dose-response, random and non-J controls, off-target metrics, generation
quality checks, and stability over seeds or prompt samples.

## Inputs

- [`configs/qwen35_4b.yaml`](configs/qwen35_4b.yaml) is the formal GoEmotions
  configuration. It pins the Qwen model/tokenizer, seven binary concepts,
  layers, BOS policy, probe search, and a local float32 J-lens refit.
- [`data/go_emotions_7concept_full_allowlist.json`](data/go_emotions_7concept_full_allowlist.json)
  pins the official GoEmotions revision and Parquet hashes. The formal dataset
  retains every unique, split-safe source row, including multilabel rows, and
  defines seven one-vs-rest binary targets. The older
  [`data/go_emotions_7concept_allowlist.json`](data/go_emotions_7concept_allowlist.json)
  defines a 2,000-example-per-concept balanced control only.
- [`data/builtin_abstract_concepts.jsonl`](data/builtin_abstract_concepts.jsonl)
  is a human-written 96-row offline test fixture for honesty, justice,
  compassion, uncertainty, creativity, and power. No Qwen GPU experiment
  config points to it.
- [`data/axbench_abstract_allowlist.json`](data/axbench_abstract_allowlist.json)
  is deliberately unverified and fails before remote loading. See the
  [data guide](docs/data.md) before enabling any remote concept.

The built-in data has eight positive and eight negative examples per concept,
with both labels present in train, validation, and test. It is designed to
exercise the full contract, not to establish broad semantic generalization.

The concept data has no separate vocabulary table. Every labeled text is
encoded by the exact tokenizer revision recorded in the model config. J-lens
token directions use rows of the same model revision's output unembedding. If
the model head has padded rows beyond `len(tokenizer)`, alignment takes a
zero-copy view of exactly tokenizer IDs `[0, len(tokenizer))`; it fails if the
tokenizer is larger than the head and records both row counts.

## Run

From the repository root, install the appropriate environment:

```bash
uv sync --extra dev
Concept_intervention/scripts/smoke_offline.sh
```

The offline smoke forces Hugging Face offline mode, invokes CLI config/data
validation, and runs the synthetic concept/alignment workflow tests.

For a real tiny-model integration after installing the optional stack:

```bash
uv sync --extra dev --extra llm
Concept_intervention/scripts/run_tiny_integration.sh
```

The tiny checkpoint is downloaded on first use. With a populated cache, enforce
offline execution with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

Run the pinned Qwen3.5-4B capture/probe/alignment workflow inside an existing
GPU allocation:

```bash
Concept_intervention/scripts/run_qwen35_4b.sh
```

Or submit the Quest template, which requests one A100 under account `p32737`:

```bash
sbatch Concept_intervention/scripts/run_qwen35_4b.slurm
```

The Bash launcher executes:

```bash
jlens-workspace doctor
jlens-workspace config validate Concept_intervention/configs/qwen35_4b.yaml
jlens-workspace data validate --config Concept_intervention/configs/qwen35_4b.yaml
jlens-workspace lens fit Concept_intervention/configs/qwen35_4b.yaml
jlens-workspace concept run Concept_intervention/configs/qwen35_4b.yaml --jobs 8
```

For checkpointed inspection, use `concept capture`, `concept fit-probes`, and
`concept align` separately. Set `CONCEPT_CONFIG` for Slurm or pass a config path
as the first argument to `run_qwen35_4b.sh`. Never point a rerun at an existing
scientific output; copy the YAML and give it a new `experiment_name` and
`output_dir`.

The current formal run stops after probe/J-direction alignment. The
generation-time hook exists in `jlens_workspace.interventions`, but no
intervention is run by this launcher; do not report alignment as causal
steering evidence.

## Outputs

The default Qwen config writes beneath
`artifacts/concept_intervention/qwen35_4b_go_emotions_7concept_full_ovr_v2/`.
The locally refitted lens is written separately to
`artifacts/lenses/qwen35_4b_go_emotions_7concept_full_block0_fp32.pt`. Expected stages
include:

- `manifest.json` and `run.json`: immutable pins, config hash, package versions,
  stage boundaries, and output index;
- `activations/`: one last-token activation per source text, memory-mappable
  `layer_XX.npy`, an `[N,7]` binary label matrix, concept columns, row metadata,
  and capture provenance;
- `probes/`: float64 probe vectors in original `resid_post` coordinates plus
  CV/validation/test metrics and an activation-identity manifest;
- `alignment/`: signed cosine-ranked token IDs/tokens, matched-norm orthogonal
  random controls, and exact probe/lens identities for semantic inspection;
- `interventions/`: prompts, generations, strengths, target/off-target scores,
  and matched controls when the intervention stage is run.

Treat the config and each artifact's metadata/manifest as a unit. A probe or
lens must never be reused after changing the model revision, tokenizer, BOS
policy, hook coordinate, or coordinate-changing model wrapper.

## Recorded runs

- [Qwen3.5-4B GoEmotions concept baseline (v1)](reports/qwen35_4b_goemotions_v1.md)
  records the completed probe/alignment run, held-out metrics, provenance, and
  limitations without committing the 3.2 GB artifact.
- [Interactive experiment report](../reports/index.html#concept) renders the
  registered Concept runs and SVG plots from lightweight summaries.

Shared implementation belongs in `../src/jlens_workspace/`; this directory owns
only direction-specific data, configs, launchers, and research reports. The
common evidence gates are in [the experiment protocol](../docs/experiment_protocol.md).
