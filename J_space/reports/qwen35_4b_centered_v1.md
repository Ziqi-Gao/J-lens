# Qwen3.5-4B centered J-space baseline (v1)

Status: completed exploratory baseline. This report describes the spectrum of
the centered, RMSNorm-weighted matrix and does not claim a prompt-stable or
universally low-rank J-space.

## Research question

How does the spectrum and effective dimension of
`A_l = U_eff J_l` change with source-layer depth? This run is the first
full-vocabulary centered baseline, not a multi-seed replication.

## Run identity

| Field | Value |
| --- | --- |
| Slurm job | `6739748`, `COMPLETED (0:0)`, elapsed `19:19:55` |
| Artifact | `artifacts/j_space/qwen35_4b_centered` (1011 MB, excluded from Git) |
| Manifest SHA-256 | `87821386470f0b4fa81e82851aa9fd963391a9896219f1cfd8714d487715ee0d` |
| Metrics SHA-256 | `6052255261a35352b5a8c60a4700ad66746252fdd8c22f38772e526187da5f32` |
| Config SHA-256 | `e1a26194a0e9f66185543dba13a323ed93b633c074f7187f31f08f021eac6863` |
| Model/tokenizer | `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Matrix | centered `A_l = U_eff J_l`, shape `[248077, 2560]` |
| Convention / coordinate | `rmsnorm_weighted`; block output, `resid_post` |
| Source layers / target | 0, 4, 8, 12, 16, 20, 24, 28, 30 / target layer 31 |
| Lens fit | 1,000 prompts, float32 storage, `force_bos=false` |
| Fit-prompt SHA-256 | `0cf0f8386ea3b683a45340068e90a379566bc208fbd9cd4de2e354a82ed50420` |
| Vocabulary processing | 248,077 tokenizer rows, chunk size 4,096 |
| Numeric precision | float32 operator/Jacobian; float64 Gram accumulation and eigendecomposition |
| Rank tolerance | relative `1e-7`, absolute `0` |
| Seed | 42 |
| Recorded Git commit | `null` |

The implementation streamed vocabulary chunks and accumulated the
`[2560, 2560]` Gram matrix; it did not materialize the production
`[248077, 2560]` matrix.

## Spectrum and effective dimension

| Layer | Numerical rank | Entropy rank | Participation ratio | Stable rank | k90 | k95 | k99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2560 | 85.35 | 44.37 | 12.11 | 83 | 118 | 325 |
| 4 | 2560 | 122.12 | 67.42 | 16.90 | 114 | 161 | 343 |
| 8 | 2560 | 221.95 | 107.28 | 21.58 | 229 | 320 | 583 |
| 12 | 2560 | 334.06 | 160.13 | 30.62 | 359 | 512 | 896 |
| 16 | 2560 | 501.65 | 189.83 | 27.58 | 606 | 846 | 1356 |
| 20 | 2560 | 1062.75 | 493.28 | 55.50 | 1132 | 1446 | 1969 |
| 24 | 2560 | 1505.98 | 998.72 | 119.46 | 1401 | 1708 | 2150 |
| 28 | 2560 | 1745.66 | 1242.44 | 147.53 | 1613 | 1893 | 2249 |
| 30 | 2560 | 1947.95 | 1338.14 | 130.43 | 1818 | 2051 | 2324 |

Here `k90`, `k95`, and `k99` are the minimum numbers of right-singular
directions required to capture the corresponding fraction of
`sum(singular_value**2)`.

Every layer is full numerical rank at the recorded tolerance. The meaningful
structure is therefore energy concentration rather than exact rank
deficiency: layer 0 needs only 83 directions for 90% energy, while layer 30
needs 1,818. Both the entropy rank and participation ratio show a strong
early-to-late expansion.

## Layer-to-layer subspace drift

A read-only post-run audit compared the first 64 right-singular directions at
each layer using `||B_i.T @ B_j||_F^2 / 64`. Dimension matching avoids the
trivial overlap increase caused by larger energy bases.

| Adjacent layers | Top-64 overlap |
| --- | ---: |
| 0 → 4 | 0.7613 |
| 4 → 8 | 0.5647 |
| 8 → 12 | 0.6231 |
| 12 → 16 | 0.7536 |
| 16 → 20 | 0.5241 |
| 20 → 24 | 0.5307 |
| 24 → 28 | 0.5512 |
| 28 → 30 | 0.6875 |

Adjacent layers retain moderate-to-high overlap, while layer 0 versus layer 30
has overlap `0.0247`. The dominant subspace is locally continuous but is
substantially reorganized across the full network depth.

## Interpretation and limitations

- The supported descriptive result is progressive dimensional expansion:
  early layers are strongly anisotropic and energy-concentrated, while late
  layers distribute sensitivity over many residual directions.
- The data do not support the statement that J-space is exactly low rank at
  any analyzed layer.
- Only one centered run and one fit-prompt sample were analyzed. Uncentered,
  row-normalized, tolerance-sensitivity, and independent prompt-sample
  replications were not completed in v1.
- `git_commit`, `lens_revision`, and top-level dataset identity are null. The
  lens metadata records the prompt path and hash, but the artifact is not fully
  attributable to an exact source checkout.
- The concept v1 run used a different fit-prompt hash. Its probe projections
  into these bases are exploratory and cannot establish preferential J-space
  enrichment without matched random-subspace controls.
