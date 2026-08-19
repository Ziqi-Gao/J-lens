# Qwen3.5-4B three-method pointwise LLM judge v2

Status: pipeline complete; calibration passed; final analysis invalid under the
registered guardrails. This is a post-hoc, descriptive-only comparison and not
confirmatory evidence that any intervention method is superior.

The machine-readable companion is
[`qwen35_4b_three_method_pointwise_judge_v2.summary.json`](qwen35_4b_three_method_pointwise_judge_v2.summary.json).

## Study identity

| Field | Value |
| --- | --- |
| Experiment | `qwen35_4b_three_method_pointwise_judge_v2` |
| Protocol / design | `concept_intervention_llm_judge_pointwise_v2` / `three_method_pointwise_judge_v2` |
| Parent generation suite | `qwen35_4b_three_method_intervention_v1` |
| Methods | J-component, RAPTOR, ITI-native mass-mean |
| Concepts | seven GoEmotions concepts, equal-weight macro aggregation |
| Judges | `mistralai/mistral-small-2603`, `anthropic/claude-haiku-4.5` |
| Tasks | 4,480 per judge per split; validation and test; 17,920 calls total |
| Missing / provider-filtered | 0 / 0 |
| Test analysis rows | 4,480 two-judge averaged pointwise rows |
| Registered budget / observed cost | USD 22.00 / USD 18.50969747 |
| Durable artifact size | 305 MB, excluded from Git |
| Completion time | 2026-08-15 |

The study is a pointwise-only adaptation registered after the parent v1 failed
its pairwise order-consistency gate. It reused no parent response. Method
selection used validation only; test judgments were not used to choose steering
configurations. Inference clusters all decodings by prompt ID and macro-averages
the seven concepts equally.

## Calibration and completion

Both judges completed all 4,480 validation tasks. The registered pointwise gate
passed: target-expression agreement had Spearman `0.7037`, MAE `13.7576`,
`60.63%` within 10 points, and `80.07%` within 20 points. Pairwise gates were
explicitly not applicable in this v2 design. Passing calibration unlocked test;
it did not guarantee that final output-quality guardrails would pass.

Across validation and test, the run consumed 17,637,045 prompt tokens and
2,225,237 completion tokens. Provider-reported or registered costs summed to
USD `18.50969747`, below the approved USD 22 cap.

## Primary test conditions

Scores are means over 448 test rows per role after averaging the two judges.
Quality is the transparent mean of coherence and relevance. Invalid and refusal
are rates.

| Role | Target expression | Coherence | Relevance | Quality | Invalid | Refusal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Common baseline | 21.69 | 72.81 | 81.95 | 77.38 | 1.90% | 0.00% |
| J-component | 30.93 | 5.88 | 8.17 | 7.02 | 60.04% | 0.00% |
| RAPTOR | 14.75 | 21.69 | 21.78 | 21.73 | 33.93% | 0.22% |
| ITI-native | 15.07 | 6.09 | 6.64 | 6.36 | 72.88% | 0.00% |

All three selected interventions show severe language-quality degradation
relative to the common zero-intervention baseline. The registered overall
invalid limit was 2%; the observed all-condition rate was 33.34% (24.40% for
Mistral and 42.28% for Claude). The invalid guardrail therefore failed.

## Registered primary contrasts

Differences are treatment minus reference. Confidence intervals are 95%
prompt-cluster bootstrap intervals.

| Contrast | Target expression (95% CI) | Quality (95% CI) | Quality gate | Refusal gate |
| --- | ---: | ---: | --- | --- |
| J - RAPTOR | +16.18 [13.53, 18.72] | -14.71 [-15.92, -13.53] | fail | pass |
| J - ITI-native | +15.86 [14.04, 17.64] | +0.66 [-0.57, 1.87] | pass | pass |
| RAPTOR - ITI-native | -0.32 [-2.43, 1.97] | +15.37 [14.01, 16.74] | pass | pass |

J-component received higher target-expression scores than the two other
interventions, but it was materially worse than RAPTOR on quality. RAPTOR and
ITI-native were indistinguishable on the macro target-expression contrast,
while RAPTOR had higher quality. These are descriptive patterns only: the
global invalid-output guardrail and one primary quality guardrail failed, so
`analysis_valid=false` and confirmatory interpretation is prohibited.

## Cross-judge agreement and audit package

On the 4,480 test rows, judge agreement was strong in rank but less exact in
absolute score:

| Field | Spearman | MAE | Within 20 |
| --- | ---: | ---: | ---: |
| Target expression | 0.6883 | 13.28 | 80.94% |
| Coherence | 0.9302 | 14.36 | 72.99% |
| Relevance | 0.8852 | 12.13 | 78.55% |

Exact agreement was 81.85% for invalid and 99.96% for refusal. The artifact
also exports 56 deterministic, method-blind human-audit tasks, eight per
concept. The checked-in summary records their hashes but intentionally does not
publish prompts, generations, blank annotation rows, or private unblinding
data. No completed human labels were registered in the calibration gate.

## Provenance and limitations

| Artifact | SHA-256 |
| --- | --- |
| Frozen config | `a2f1a29a768009b2cd9f1222d35b1c79e430b70047775eb18c9d088e78413d8a` |
| Manifest | `d8ba46c2d9146e922023fe196917f1d38c8807ab8cecbc1261dda23c7991554c` |
| Calibration | `37f8995986071d157b325603561d3ee5c25dfdec7f5dbded0337834956e5cba9` |
| Final report | `c4eee6e0f91521dba9ee1d3f9de78f65de13617a7e57f9e18f2448cc93e82bf2` |
| Unblinded analysis rows | `b512e3b6be7a26152d0f31ca8a33ded20f36ce87b2c352979e926dd00d43e674` |
| Aggregate response artifacts | `fd3b70aa00c5e2a27604f65cdeaa44f4febd2259966c920cbaa888c9004c048b` |
| Rubric | `6060712756f74814fbef87b15fd98cfec2bb33dfc66a108d4b211cac7d02f21b` |
| Human-audit manifest | `563cb036424449e5df0b1cb434c5b6b8c4e5da0d945d3849117b87ebd6b3e8a4` |

The authoritative artifact remains below
`$JLENS_DATA_ROOT/runs/qwen35_4b_three_method_intervention_v1/artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/llm_judge_three_method_pointwise_v2`.
It contains the sealed tasks, private maps, per-call responses, calibration,
human-audit package, unblinded rows, report, and indexes. Those generated
artifacts are immutable and excluded from Git. This report must not be used to
reconstruct or replace them.
