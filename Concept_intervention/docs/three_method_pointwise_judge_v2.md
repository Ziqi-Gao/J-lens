# Three-method pointwise LLM judge v2

## Status and scope

This is a new, isolated post-calibration adaptation of the three-method
LLM-as-a-Judge experiment. It does not modify or unlock
`qwen35_4b_three_method_llm_judge_v1`. The parent v1 completed all 14,336
validation judgments and failed its preregistered pairwise order-consistency
gate; its test responses remained at zero.

The v2 study is descriptive-only. It registers pointwise judgments for
J-component, RAPTOR, and ITI plus the same baseline and sensitivity controls.
Pairwise judgments, arbitration, expert-review API calls, and confirmatory
interpretation are outside scope.

## Registered design

- Protocol: `concept_intervention_llm_judge_pointwise_v2`
- Study: `three_method_pointwise_judge_v2`
- Primary judge: `mistralai/mistral-small-2603`
- Secondary judge: `anthropic/claude-haiku-4.5`
- Splits: validation, then test only after calibration passes
- Tasks per judge per split: 4,480 pointwise tasks
- Base formal calls: 17,920
- Parent-response reuse: none
- Human audit: eight deterministic blind test tasks per concept
- Output: `llm_judge_three_method_pointwise_v2`

The v2 task IDs use the new protocol version, and its manifest seals the failed
parent manifest, calibration, and repair outcome. Preparation fails if a parent
seal changes or if any parent test response appears.

## Gates and interpretation

Test access requires complete new validation responses and all applicable
pointwise gates:

- minimum completion rate 0.98;
- target-expression Spearman correlation at least 0.60;
- target-expression mean absolute error at most 20;
- the human gate only if an annotation file is explicitly registered.

The pairwise gates are recorded as not applicable, not silently removed from
the parent. A v2 report always identifies the adaptation as post hoc and
descriptive. It estimates the three registered primary pointwise contrasts and
reports coherence/relevance quality, refusal, invalid-rate, and cross-judge
agreement separately.

## Budget and launch lock

The configuration sets a hard experiment cap of $22 and preserves $25 of
OpenRouter credit. The exact expected token usage and cost are computed from
the frozen task files before launch. Formal API execution remains locked until
an operator creates the config-and-manifest-bound approval file after reviewing
that report. The two one-task smoke calls, if authorized, remain isolated from
formal responses.

The controller is
`Concept_intervention/scripts/run_three_method_pointwise_judge_v2_local.sh`.
It must run from an immutable snapshot under
`/scr/del6500/J-lens/runtime/llm-judge/three-method-pointwise-v2/snapshots/`.
