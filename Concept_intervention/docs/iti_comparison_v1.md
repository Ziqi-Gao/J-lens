# Qwen3.5-4B ITI comparison v1

## Question

How does the registered sparse J-component intervention compare with
Inference-Time Intervention (ITI) when the model, tokenizer, GoEmotions rows,
candidate labels, prompt bank, and held-out evaluation templates are fixed?

This is a reproduction-style baseline, not a claim that ITI's published
TruthfulQA hyperparameters transfer to GoEmotions.

## Upstream algorithm

The method follows the
[ITI paper](https://arxiv.org/abs/2306.03341), and its core is adapted from
[`likenneth/honest_llama`](https://github.com/likenneth/honest_llama) at commit
[`2c6b217`](https://github.com/likenneth/honest_llama/tree/2c6b2179be7b5aa8f0a171688cf9e01b812ca327).
The upstream MIT license is kept in
[`third_party/honest_llama/LICENSE`](../../third_party/honest_llama/LICENSE), and
the attributed core is isolated in
[`src/jlens_workspace/_vendor/honest_llama_core.py`](../../src/jlens_workspace/_vendor/honest_llama_core.py).

The following algorithmic steps retain the original implementation and
semantics:

1. train one default-regularized scikit-learn logistic regression per head;
2. rank heads by validation accuracy;
3. use the positive-minus-negative mass-mean direction as the primary method;
4. normalize each head direction and scale it by the standard deviation of
   development activations projected onto that direction;
5. add `alpha * projection_std * unit_direction` to the final token on every
   autoregressive forward call.

The probe-weight direction is retained as a sensitivity analysis. Five
original-style random-head/random-direction controls are evaluated at the
selected K.

## Necessary Qwen/data adaptations

The model and tokenizer are both `Qwen/Qwen3.5-4B` at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The dataset remains the existing
53,861-source GoEmotions v2 artifact at upstream revision
`add492243ff905527e67aeb8b80c082af02207c3`; no new split is created.

Qwen3.5-4B is a hybrid model. The faithful primary baseline includes only its
eight ordinary full-attention layers `[3, 7, 11, 15, 19, 23, 27, 31]`. Each
captured boundary is the input to `self_attn.o_proj`, with shape
`[N, H=16, D_head=256]`. The 24 linear-attention blocks do not expose the same
head coordinate and are excluded rather than silently treated as MHA heads.

This coordinate is intentionally different from the J-lens experiment's
`resid_post [N, D_model=2560]` coordinate. It is a named, method-specific
exception needed to reproduce ITI: moving the addition to block output would
change the published intervention. Results must therefore be described as a
comparison between methods, not as two directions injected at the same model
boundary.

The two method pipelines are also separate in code. J-component direction
loading and `resid_post` hooks live in the J workflow and intervention module;
ITI head fitting and pre-`o_proj` hooks live in the ITI workflow and module.
Neither method workflow imports the other. They share only the method-neutral
prompt/candidate-label evaluation contract, the pinned model/tokenizer, and
identity-checked source data and split artifacts.

The original TruthfulQA grouped answer layout is replaced by the existing
GoEmotions `[N,7]` label matrix. Within the existing train and validation
splits, each concept's positive and negative rows are seed-balanced so that the
original accuracy-based head ranking is meaningful for a one-vs-rest target.
Test rows are never used to train probes, compute mass means, rank heads, or
estimate projection standard deviations.
Because GoEmotions has no separately registered unlabeled `tuning_activations`
bank analogous to the upstream TruthfulQA generation bank, projection standard
deviations are estimated on the same balanced train+validation development
activations used for directions. The exact source-row indices are saved; no
test activation enters this estimate.

## Selection and evaluation

Two independent held-out boundaries are enforced:

- Data fitting: only dataset `train` and `validation` rows fit ITI directions;
  dataset `test` rows used for fitting is recorded as zero.
- Prompt selection: the `choose_*` and `complete_*` templates select K from
  `[4, 8, 16, 32, 48]` and a positive alpha from the signed sweep. The
  `classify_*` and `report_*` templates are then used once for comparison.

The J-component reference is reanalyzed with the exact same prompt split: its
positive strength is selected on the validation templates, and its effect is
reported on the test templates. Raw alpha values are not compared across
methods because their coordinates and normalization differ. The primary value
is instead the held-out change in target margin relative to that method's
zero-strength baseline. Signed dose-response curves, off-target candidate
probabilities, the probe-weight ITI variant, and five random controls are also
saved. The comparison additionally reports a deterministic 10,000-resample
paired prompt bootstrap 95% interval for each method's selected held-out
effect; the 14 templates remain the unit of inference, so these intervals do
not establish broad prompt-distribution generalization.

No LLM-as-judge call is required for this registered comparison. The outcome
is a deterministic one-token choice among seven fixed labels, so candidate
log-probability, normalized candidate probability, target margin, and rank are
direct measurements. Four-token greedy generations are retained only for
manual audit. A later open-ended experiment would need a frozen classifier,
blind LLM judge, human audit subset, and fluency/KL metrics under a separately
registered protocol.

## Run and artifacts

The immutable configuration is
[`configs/qwen35_4b_iti_comparison_v1.yaml`](../configs/qwen35_4b_iti_comparison_v1.yaml).
Submit the four-stage dependency chain from a clean committed checkout:

```bash
Concept_intervention/scripts/submit_qwen35_4b_iti_comparison_v1.sh
```

If the existing J-component array is still running, export its terminal Slurm
job ID as `J_INTERVENTION_JOB_ID` before submission so the final comparison
waits for both experiments. The final index is fail-closed if the reference J
index is absent or incomplete.

Expected paths under
`artifacts/concept_intervention/qwen35_4b_iti_comparison_v1/` are:

- `head_activations/`: eight float32 `[53861,16,256]` arrays (about 7.1 GB),
  `[53861,7]` labels, row identity, and model/data provenance;
- `directions/<concept>/`: ranked mass-mean, probe-weight, and random direction
  files plus selection-only metrics;
- `targets/<concept>/`: validation scores, held-out test scores, audit
  generations, and a summary;
- `index.json`: completeness and content hashes;
- `comparison.json`: identity-checked ITI versus J held-out effects.

GPU results are not implied by the presence of these launchers; report them
only after the artifact index and comparison are complete.
