# Qwen3.5-4B concept-vector J-space occupancy pilot (v1)

Status: completed pilot under `concept_occupancy_method_v1`. This report is
descriptive and correlational; it makes no causal steering claim and does not
claim to reproduce Anthropic's exact occupancy implementation (their atom
normalization, random-control construction, and crossing rule remain
unconfirmed, so every methodological choice below is explicitly versioned).
Per reviewer decision, BOTH crossing rules are reported side by side; neither
is designated primary.

## Research question

How many token-facing J-lens directions does a sparse non-negative
combination need before its marginal reconstruction gain on a probe-derived
concept direction stops exceeding a matched random-dictionary control — and
does that concept-specific sparse complexity grow with layer the way the
global J-space spectral dimension does?

## Run identity

| Field | Value |
| --- | --- |
| Slurm jobs | shards `7364458/7364459/7364460` `COMPLETED` (86–105 min each); prior shards `7324698/7324699/7324700` `TIMEOUT` at 2 h after completing the full rmsnorm_weighted half; first-attempt shards failed at startup (safetensors tensor naming, fixed in `e52645c`); superseded monolithic `7322965` cancelled while pending |
| Artifact | `artifacts/concept_intervention/qwen35_4b_concept_occupancy_pilot_v1` (72/72 combos) |
| Git commit / branch | `e52645c` on `run/concept-occupancy-v1` (clean tree) |
| Model/tokenizer | `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; `resid_post`; `force_bos=false` |
| Dictionary lens | `qwen35_4b_jspace_9layers_goemotions_full_block0_fp32.pt`, SHA-256 `cdb356a078f28f5c…` (the same lens behind the v2 global spectrum runs; NOT the concept-lane lens) |
| Dictionary | UN-CENTERED `A_l = U_eff J_l`; `[248077, 2560]` streamed, never materialized; float64 |
| Probe targets | v2 raw-coordinate logistic probes (GoEmotions full OVR v2, `add492243f…`), per-vector SHA-256 verified; `+w` and `-w` analyzed separately |
| Scope | concepts gratitude/optimism/approval × layers 8/20/28 × ±w × conventions {rmsnorm_weighted, raw} × selection {positive_cosine, raw_positive_dot} |
| Solver | exact full-vocabulary streaming non-negative pursuit, per-step residual re-search, NNLS refit on raw atoms, K_max=64, every integer k stored |
| Random control | `matched_gaussian_atom_norms_v1`, seeds [101, 202, 303, 404, 505], NumPy Philox chunk-deterministic |
| Crossing rules | `first_nonexceed_v1` (A) and `consecutive3_nonexceed_v1` (B) vs per-k median of seed controls |
| Seed | 42 |

## Occupancy results (both rules)

Median across concepts and signs; per-combo values range 1–6 with **zero
right-censored combos** (every crossing happens by k ≤ 6).

| Convention | Layer | med K_A | med K_B | med EF@64 | med cos(w, w_J@64) | med K_90_attainable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rmsnorm_weighted | 8 | 1 | 1 | 0.021 | 0.146 | 30 |
| rmsnorm_weighted | 20 | 1 | 2 | 0.060 | 0.245 | 30 |
| rmsnorm_weighted | 28 | 2 | 3 | 0.099 | 0.314 | 38 |
| raw | 8 | 1 | 1 | 0.011 | 0.105 | 16 |
| raw | 20 | 1 | 1 | 0.036 | 0.188 | 19 |
| raw | 28 | 2 | 2 | 0.050 | 0.224 | 25 |

Rules A and B agree exactly on every rmsnorm_weighted/positive_cosine combo;
under raw_positive_dot rule B is occasionally larger by 1–3 (max K_B = 6 at
layer 28, +w). Absolute explained-fraction thresholds: 1% is reached at
k = 1–8 everywhere; 5% only at layers 20/28 (k = 5–25); 10% only for
optimism at layer 28 (k = 35); 20% is reached nowhere.

## Findings

1. **Occupancy is tiny.** Only the first 1–6 token-J atoms beat the
   norm-matched random control; afterwards real token directions add no more
   than random directions. `+w` sides at layer 28 carry the largest values
   (3–6); `-w` sides sit at 1–2 with semantically incoherent supports —
   the non-negative cone asymmetry is real and the ±w separation necessary.
2. **The layer trend is monotone but very mild.** K, EF@64, and cosine all
   rise with depth in both conventions, yet across the same layers the global
   centered spectrum's k90 expands 229 → 1132 → 1613. Global dimensional
   expansion translates into only a units-scale increase in concept-specific
   sparse complexity. (Three layers; descriptive only.)
3. **Explained fractions land in Anthropic's reported band.** EF@64 of
   6–10% at layers 20/28 (primary convention) matches the workspace paper's
   6–7% (concept vectors) and 10–15% (probes) J-component variance shares.
4. **Selected atoms are semantically right where probes are strong.**
   gratitude+: ` thank` (first pick, L20), ` appreciate`, ` Geschenk`, `致谢`
   (L28); optimism+: a multilingual hope cluster (`Hope`, `好运`, `也希望`,
   ` امید`, ` hopefully`, `顺利`). approval+ supports remain incoherent —
   consistent with v1's finding that the approval probe/alignment is
   unreliable. Selection is stable across conventions and selection modes.
5. **Convention sensitivity.** Paired |ΔK_B| has median 0 (max 4):
   the tiny-occupancy conclusion is convention-independent. rmsnorm_weighted
   systematically explains more than raw (ΔEF median +2.5 pp, always
   positive), supporting it as the primary frame for later comparisons.

## Interpretation caution: control strength

248,077 matched-norm random directions in 2,560 dimensions form a strong
reconstructor (expected best cosine against a fixed residual ≈ 0.10, the same
order as real token atoms' 0.05–0.28). Small K_occ therefore conflates "few
informative token directions" with "very strong null". A lower-cardinality
control (fraction ≈ 1/16 of V) is scheduled for the formal run to separate
the two readings. Until then, K_occ values are lower-bound-flavored and rule-
and control-relative by construction.

## Limitations

- Method-v1 choices (selection scores, NNLS refit cadence, control
  construction, crossing rules) are project-defined, not confirmed against
  Anthropic's implementation; results are labeled accordingly.
- Single probe replicate per concept (schema supports replicates; bootstrap
  stability is future work). Three concepts, three layers.
- The greedy pursuit has no exact-recovery guarantee over a highly coherent
  248k-atom dictionary; K values are algorithm-relative.
- `w_nonJ(k)` is a sparse-reconstruction residual, not an orthogonal
  complement.

## Next

Formal run (approved): 7 concepts × 6 shared layers × ±w × both conventions
× both selection modes, plus the low-cardinality random-control sensitivity;
per-layer 2 h gengpu shards (never gengpu-long). Causal K_causal experiments
remain a separate, later stage.
