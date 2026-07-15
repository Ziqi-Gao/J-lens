# Qwen3.5-4B GoEmotions concept baseline (v1)

Status: completed exploratory baseline. This report records probe and token-J
alignment evidence; it does not claim causal steering.

## Research question

Can seven GoEmotions concepts be decoded from Qwen3.5-4B `resid_post`
activations, and do the resulting raw-coordinate probe directions retrieve
semantically related token J-directions? This run is the first full-scale
baseline, not a controlled replication.

## Run identity

| Field | Value |
| --- | --- |
| Slurm job | `6741193`, `COMPLETED (0:0)`, elapsed `20:45:32` |
| Artifact | `artifacts/concept_intervention/qwen35_4b_go_emotions_7concept_full_ovr_v1` (3.2 GB, excluded from Git) |
| Manifest SHA-256 | `c86b1b3ebeed162f8788feebe3f5026f0b4b9539e27f704e6475728098a86f7a` |
| Config SHA-256 | `c95a286bad850d09a6b88812a96099b5b61682ce4fa8225580db9b71b4f362a2` |
| Model/tokenizer | `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Dataset | `google-research-datasets/go_emotions` at `add492243ff905527e67aeb8b80c082af02207c3` |
| Dataset hash | `sha256:dd4d513e966dd4817dbf668e19589cc75e23a6b4ec94b030a2959f450fb7c9ea` |
| Coordinate | transformer block output, `resid_post`, dimension 2560 |
| Source layers / target | 8, 12, 16, 20, 24, 28 / target layer 31 |
| Lens fit | 1,000 prompts, float32 storage, `force_bos=false` |
| Fit-prompt SHA-256 | `4dcb7c85ecbcbaade5fd029019b8f03d9ed373deeb648981aa75afc98e5eaa6e` |
| Seed | 42 |
| Recorded Git commit | `null` |

The capture contains six `[53861, 2560]` activation arrays and one
`[53861, 7]` label matrix. The original one-vs-rest preparation contained
377,027 task rows. Source-text split sizes are 43,112 train, 5,370 validation,
and 5,379 held-out test examples.

Probe `C` was selected using five-fold stratified-group CV on training data.
The selected model was refit on train plus validation and evaluated once on
the held-out test split. Saved float64 coefficients are mapped back from
standardized features into original residual coordinates.

## Held-out probe results

Values aggregate the six source layers. AP is reported because the concepts
are imbalanced.

| Concept | Mean test ROC AUC | Layer range | Mean test AP | Best layer |
| --- | ---: | ---: | ---: | ---: |
| admiration | 0.8815 | 0.8771–0.8874 | 0.5203 | 28 |
| approval | 0.7851 | 0.7780–0.7951 | 0.2476 | 20 |
| curiosity | 0.9260 | 0.9127–0.9329 | 0.4384 | 20 |
| disapproval | 0.8506 | 0.8460–0.8576 | 0.2454 | 20 |
| gratitude | 0.9667 | 0.9643–0.9717 | 0.8452 | 28 |
| love | 0.9217 | 0.9077–0.9355 | 0.4930 | 28 |
| optimism | 0.8833 | 0.8772–0.8895 | 0.3304 | 20 |

Mean test ROC AUC across concepts is highest at layer 20 (`0.8927`), followed
by layer 28 (`0.8913`). A read-only audit recomputed all 42 raw-vector test
AUCs directly from saved activations and labels; the maximum absolute mismatch
from the recorded metrics was `0.0`.

## Token-J alignment

Positive layer-28 retrievals are semantically clear for several concepts:
`Excellent`/`superb` for admiration, `appreciate`/`appreciated` for gratitude,
`love`/`Love` for love, and `Hope`/`hope` for optimism. Curiosity retrieves
`答案`/`answers`, while disapproval is dominated by `nor`; these two may encode
question/answer and negation shortcuts. Approval retrieves mixed multilingual
or fragmentary tokens and is not a reliable semantic alignment result.

Across all 42 alignments, the largest positive cosine has mean `0.1122` and
range `0.0501–0.2834`. Cross-layer top-10 token stability is strongest for
optimism, gratitude, disapproval, and love, and weakest for approval.

## Interpretation and limitations

- The held-out probe result is numerically reproducible and establishes
  correlational linear decodability, especially for gratitude, curiosity, and
  love.
- The v1 alignment files contain neither matched random controls nor J/non-J
  decomposition (`decomposition=null`). The intervention stage was not run.
  Alignment must not be described as causal steering.
- `git_commit` and `lens_revision` are null. The repository pins the official
  Jacobian-lens source, but this artifact alone cannot prove the exact source
  snapshot used at execution time.
- The J-space v1 run used a different lens-fit prompt artifact. Cross-direction
  projection from these two v1 runs is exploratory rather than a controlled
  comparison.
- Approval should not advance to intervention. Curiosity and disapproval need
  shortcut-controlled datasets. A matched-provenance v2 rerun with random and
  non-J controls is required before causal experiments.
