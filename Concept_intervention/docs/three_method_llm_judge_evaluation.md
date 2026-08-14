# Three-method LLM-as-a-Judge evaluation

This lane is the post-hoc, descriptive evaluation of the sealed
`qwen35_4b_three_method_intervention_v1` generations. It is a new study, not an
extension of the J-component/RAPTOR pilot. The old v16 pilot is read only and is
used solely for an explicit reuse assessment.

## Registration and isolation

- Experiment: `qwen35_4b_three_method_llm_judge_v1`
- Config: `Concept_intervention/configs/qwen35_4b_three_method_llm_judge_v1.yaml`
- Output: `.../llm_judge_three_method_v1`
- Runtime: `/scr/del6500/J-lens/runtime/llm-judge/three-method-v1`
- Controller: `run_three_method_llm_judge_v1_local.sh`

Preparation verifies the sealed comparison index, comparison payload, all three
method indexes, every selected generation shard, and all three candidate-rescore
indexes/manifests. It fails if the three zero-intervention generation streams are
not identical. It never writes to a generation, rescore, method index, or
comparison artifact.

The prepared manifest seals the source registry, code, rubric, source config,
frozen config, public tasks, private maps, selections, and legacy reuse
assessment. Existing registered artifacts are immutable: a repeated prepare must
match byte-for-byte semantic content.

## Conditions and selection

All method configuration is selected from `candidate_score_rescore_v1`
validation rows. Test judgments never participate in configuration selection.

Primary pointwise roles are:

- common zero-intervention baseline;
- validation-selected positive J-component intervention;
- validation-selected RAPTOR target probability above 0.5;
- validation-selected ITI native global-head ranking with mass-mean direction.

Secondary pointwise sensitivity/control roles are:

- J full, non-J, and matched-random seed 101;
- ITI layer-matched mass-mean, native probe-weight, and matched-random seed 101.

The primary pairwise matrix contains all three unordered method pairs:

- J-component versus RAPTOR;
- J-component versus ITI-native;
- RAPTOR versus ITI-native.

Every pair is presented in both A/B orders. Public tasks contain no method,
condition, strength, Top-K, variant, random seed, file path, hypothesis, or
provenance. Private maps are mode 0600 in a mode 0700 directory.

## Fixed task matrix

Each condition uses seven concepts, 16 open prompts per split, greedy decoding,
and sample seeds 1001, 2002, and 3003.

| Task set | Per split | Both splits | Per two base judges |
| --- | ---: | ---: | ---: |
| Pointwise: 10 roles | 4,480 | 8,960 | 17,920 |
| Pairwise: 3 pairs × 2 orders | 2,688 | 5,376 | 10,752 |
| Base total | 7,168 | 14,336 | 28,672 |

Arbitration and expert-review counts are selective and cannot be known until both
base judges finish test. They never alter the primary two-judge estimator.

## Judge and request policy

The registered base judges are `mistralai/mistral-small-2603` and
`anthropic/claude-haiku-4.5`. The independent Mistral and Anthropic model
families preserve cross-judge diversity while keeping the registered study
within its fixed USD 50 hard cap. Arbitration is
`google/gemini-3.6-flash`; expert review is
`anthropic/claude-sonnet-5`.

Requests use temperature zero, exact model IDs, registered provider orders,
provider data collection denied, required parameters, and no cross-model
fallback. Returned model identity must equal requested identity. Scored fields
are strictly validated. Evidence and rationale tolerance cannot change scores,
preferences, or flags.

The Mistral route is pinned to the first-party `mistral` provider. The study
reserves USD 25 of live OpenRouter credit and refuses any invocation whose
conservative projection would exceed either that reserve or the USD 50 study
cap. Based on the completed v16 token profile and selective-review rates, the
central projection is about USD 41; the cap is a bound, not a spending target.

Each successful task has one atomic response file containing the blinded request
payload, raw provider response, usage/cost, provider, exact model, timestamps,
retry attempts, and validated judgment. Authorization headers and API keys are
never persisted. Failures have separate per-task event files with attempt and
error metadata. Safe resume is by validated task identity, not by a coarse batch
marker.

## Statistics and interpretation

The three primary estimands are J − RAPTOR, J − ITI-native, and RAPTOR −
ITI-native. Differences are paired by concept, prompt ID, decoding, and seed.
Bootstrap and permutation inference clusters all decodings at prompt ID, then
macro-averages the seven concepts with equal weight. The 21 concept-level primary
target-expression tests receive Benjamini-Hochberg correction.

Coherence and relevance form a transparent quality average. Quality and refusal
remain separate noninferiority/guardrail outcomes; they are never multiplied into
target expression. Pairwise reporting includes both-order consistency, position
bias, each method-pair preference, and cross-judge agreement.

This study is always `descriptive_only=true` because earlier two-method test
judgments and pilot protocols were observed. `complete=true` means the pipeline
and audit materials are complete. It does not imply that analysis guardrails
passed. `analysis_valid`, calibration, invalid, quality, and refusal status are
reported separately.

## Local preparation and smoke

These commands do not require a GPU. Validate and prepare do not require an API
key:

```bash
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=""
PYTHON=/scr/del6500/J-lens/envs/three-method/bin/python
CONFIG=Concept_intervention/configs/qwen35_4b_three_method_llm_judge_v1.yaml

"${PYTHON}" -m jlens_workspace.cli judge validate --config "${CONFIG}" --json
"${PYTHON}" -m jlens_workspace.cli judge prepare --config "${CONFIG}" --json
Concept_intervention/scripts/run_three_method_llm_judge_v1_local.sh --dry-run
```

Only one validation pointwise smoke per base judge is authorized before formal
budget approval. Load the secret by sourcing the protected file; never print or
copy it:

```bash
set -a
source /scr/del6500/J-lens/runtime/llm-judge/secrets.env
set +a
"${PYTHON}" -m jlens_workspace.cli judge run --config "${CONFIG}" \
  --role primary --task-set pointwise --split validation --limit 1 --smoke --jobs 1 --json
"${PYTHON}" -m jlens_workspace.cli judge run --config "${CONFIG}" \
  --role secondary --task-set pointwise --split validation --limit 1 --smoke --jobs 1 --json
```

## Formal run lock

All non-smoke requests require a mode-0600 approval JSON at the registered path.
It must contain `approved=true`, the experiment name, exact config SHA-256,
exact prepared-manifest SHA-256, and the exact approved maximum study cost.
Changing config, code, rubric, sources, or tasks invalidates the approval.

After explicit operator approval, create an immutable snapshot under
`.../three-method-v1/snapshots/<sha256>` and launch the controller with the
local CPU/RAM lease broker from the dedicated user service
`jlens-three-method-judge-v1.service`. Do not reuse the generation service.
The controller disables CUDA, uses an experiment budget lock, records stage
status, resumes per task, and stops after three consecutive no-progress batches.

Execution order is validation, calibration, test, arbitration, expert review,
human-audit export, and aggregation. A failed calibration stops before any
three-method test request; thresholds must not be changed to force passage.
