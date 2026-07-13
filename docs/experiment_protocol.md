# Experiment protocol

## Direction 1: abstract concept intervention

Run every selected layer and concept through the same gates:

1. Validate the dataset and freeze its SHA-256 fingerprint.
2. Capture the labeled text's final non-padding token at block output
   (`resid_post`). Do not pool a response span or a sequence.
3. Tune the L2 logistic probe's inverse penalty `C` with grouped CV on train.
4. Record validation diagnostics, then refit with fixed `C` on train+validation
   and evaluate test once.
5. Search all token-J rows in chunks with ordinary cosine. Report top positive
   and negative tokens, signs, and stability across seeds. This is the end of
   the current formal experiment.

Only after the nearest-token semantics pass review should a later experiment:

6. Optionally fit a sparse non-negative J reconstruction and keep the non-J residual.
7. Intervene with four matched directions: full probe, J component, non-J
   component, and a random orthogonal direction. Sweep signed strengths.
8. Report target concept rate, off-target concept rates, output length,
   perplexity/fluency, and dose-response. No single nearest token is treated as
   causal evidence.

The built-in dataset is an offline smoke seed. The formal GoEmotions benchmark
uses all 53,861 unique split-safe source texts for every concept, preserves
multilabel rows, and defines one-vs-rest labels. Its expanded JSONL is an audit
representation only: activation capture deduplicates by global source group and
writes one `[N,7]` label matrix, so each source text is forwarded once. A hard
50,000-example-per-concept lower bound is enforced at load time.

## Direction 2: J-space matrix analysis

For each layer, analyze three separately named objects:

- raw `W_U J_l`;
- RMSNorm-weighted `W_U diag(gamma) J_l`;
- optionally row-normalized and/or frequency-weighted variants.

Accumulate the Gram matrix in float64 over vocabulary chunks. Report numerical
rank with its tolerance, entropy effective rank, participation ratio, stable
rank, singular-value curves, and the smallest leading orthogonal bases reaching
90%, 95%, and 99% energy. Repeat the analysis on multiple fitted-lens prompt
samples before claiming a stable low-rank structure.

## Cross-direction analysis

For a probe vector `w` and the top-`r` right-singular basis `Q_r`, report

`||Q_r^T w||^2 / ||w||^2`

and the held-out AUC retained after projecting the probe into that basis. This
distinguishes a low-dimensional token-frame geometry from a concept-specific
causal direction.
