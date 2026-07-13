# J-space matrix analysis

This root-level direction studies the geometry of

\[
A_l = U_{\mathrm{eff}}J_l \in \mathbb{R}^{V\times D}
\]

at each selected transformer layer. It is a sibling of
[`Concept_intervention/`](../Concept_intervention/README.md).

This is a reproduction and extension, not the first dimensionality analysis of
the matrix. Figure 28d of the
[Workspace study](https://transformer-circuits.pub/2026/workspace/index.html)
already measured the fraction of residual dimensions needed to capture
specified shares of variance across \(W_UJ_l\). This direction adds a pinned
open-Qwen implementation with explicit numerical/effective ranks, singular
spectra, saved energy bases, normalization variants, precision metadata, and
layerwise reproducibility checks.

## Objects and conventions

`J_l` maps layer-\(l\) `resid_post` coordinates into the lens target frame.
`U_eff` is one of:

- `raw`: \(W_U\);
- `rmsnorm_weighted`: \(W_U\operatorname{diag}(\gamma)\), excluding the
  activation-dependent RMS scalar.

Here \(V=\texttt{len(tokenizer)}\). If the model output head is padded beyond
that range, the workflow analyzes a zero-copy view of rows `[0,V)` and records
both the raw head row count and effective tokenizer row count. Thus every PCA
row corresponds to a real token ID.

The configured Qwen analysis uses `rmsnorm_weighted`. The
[`qwen35_4b_row_normalized.yaml`](configs/qwen35_4b_row_normalized.yaml) variant
answers a different question by normalizing every vocabulary row before Gram
accumulation; its spectrum must not be compared as though it were the raw
matrix spectrum.

`A_l` is rectangular, so eigenvalues of `A_l` are undefined. The workflow never
materializes a production \(V\times D\) tensor. It streams vocabulary blocks,
accumulates \(G_l=A_l^\top A_l\) in float64, and reports:

- singular values from \(\sqrt{\lambda_i(G_l)}\);
- tolerance-dependent numerical rank;
- a configured relative-tolerance rank sensitivity sweep;
- entropy effective rank, participation ratio, and stable rank;
- the smallest leading right-singular bases reaching configured cumulative
  energy thresholds;
- centered mean and zero-row behavior when relevant.
- pairwise principal angles and directional coverage between layer energy
  subspaces.

PCA here is over vocabulary-row geometry. It does not by itself show that model
activations or abstract concepts occupy the same low-dimensional subspace.

## Inputs

- [`configs/qwen35_4b.yaml`](configs/qwen35_4b.yaml): uncentered,
  RMSNorm-weighted, unnormalized token rows, without total-weight
  normalization.
- [`configs/qwen35_4b_centered.yaml`](configs/qwen35_4b_centered.yaml): centered,
  RMSNorm-weighted, unnormalized token rows with total-weight normalization.
- [`configs/qwen35_4b_row_normalized.yaml`](configs/qwen35_4b_row_normalized.yaml):
  centered, RMSNorm-weighted, unit-normalized rows.

All three pin Qwen3.5-4B and refit the same nine-layer J-lens locally from
1,000 prompts, with resumable checkpoints and float32 storage. This is required
because the available 31-layer Hub artifact is float16 and cannot support a
numerical-rank claim about its discarded spectral tail. Do not substitute a
model/tokenizer revision, BOS policy, residual coordinate, or fit corpus
without producing a distinct configuration and run.

Before a stable-low-rank claim, copy the relevant YAML twice, set
`lens.fit_prompt_offset` to `1000` and `2000`, and give each copy distinct lens,
checkpoint, experiment, and output paths. Together with the default offset
`0`, these select three disjoint 1,000-prompt blocks from the pinned 3,000-prompt
artifact. Compare their spectra and subspaces; do not overwrite or mix them.

## Run

Validate local contracts without a model download:

```bash
uv sync --extra dev
J_space/scripts/smoke_offline.sh
```

The matrix numerical tests run when Torch is installed and skip explicitly in
a core-only environment.

Fit and analyze a real tiny public model after installing the optional stack:

```bash
uv sync --extra dev --extra llm
J_space/scripts/run_tiny_integration.sh
```

The first run downloads the pinned `sshleifer/tiny-gpt2`; a populated
Hugging Face cache supports strict offline mode.

Run the pinned centered PCA analysis in an existing GPU allocation:

```bash
J_space/scripts/run_qwen35_4b.sh
```

Or submit the Quest A100 template:

```bash
sbatch J_space/scripts/run_qwen35_4b.slurm
```

The launcher performs:

```bash
jlens-workspace doctor --require-llm
jlens-workspace config validate J_space/configs/qwen35_4b_centered.yaml
jlens-workspace lens fit J_space/configs/qwen35_4b_centered.yaml
jlens-workspace matrix run J_space/configs/qwen35_4b_centered.yaml
```

Analyze the row-normalized object by overriding the config:

```bash
JSPACE_CONFIG="$PWD/J_space/configs/qwen35_4b_row_normalized.yaml" \
  sbatch J_space/scripts/run_qwen35_4b.slurm
```

Every scientific rerun should use a new `experiment_name` and `output_dir`.

## Outputs

The default launcher uses the centered config and writes beneath
`artifacts/j_space/qwen35_4b_goemotions_full_centered/`:

```text
metrics.json
manifest.json
subspace_comparisons.json
layer_NNN/
├── metrics.json
├── singular_values.npy
├── eigenvalues.npy
├── basis_numerical_rank.npy
├── center_mean.npy          # centered variants only
└── basis_energy_*.npy
```

Each layer's metrics state the matrix definition, coordinate convention,
centering and normalization choices, block size, dtypes, primary rank
tolerance, tolerance-sensitivity ranks, effective-rank values, and basis
energy. The comparison file reports pairwise principal angles and both
directions of subspace coverage at each energy threshold. Interpret tail rank
only when lens storage and accumulation precision support it; a float16-saved
tail is not evidence of exact rank deficiency.

Shared implementation belongs in `../src/jlens_workspace/`; this direction owns
only configs, launchers, and reports. See the
[shared research design](../docs/design.md) for invariants.
