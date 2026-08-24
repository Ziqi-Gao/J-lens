# Qwen3.5-4B K diagnostic v1

> Historical, superseded identity. Do not launch this protocol: its target name and
> physical shard layout do not satisfy the review requirements. The immutable files
> remain only to explain old artifact identities. New work uses
> `qwen35_4b_k_diagnostic_v2.md` and the v2 configs/launchers.

`qwen35_4b_k_diagnostic_v1` is an independent, preregistered diagnostic of why concept-occupancy first crossing can occur at `K <= 4`. It does not change any prior occupancy curve, stopping rule, target selection, or artifact. It performs no generation and uses no LLM judge.

## Immutable inputs and coordinates

- Model/tokenizer: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Lens target layer: 31; source layers: 3, 7, 11, 15, 19, 23, 27.
- Target coordinate: `resid_post`.
- Token frame: uncentered rows of `U_eff J_l`, with `rmsnorm_weighted` effective unembedding.
- Solver: full-vocabulary, positive-cosine, `nonnegative_gradient_pursuit_standard`.
- Shared lens, probe manifest, row manifest, labels, seven activation arrays, and the prepared test JSONL are content-hash pinned in every config.
- Artifact root: `artifacts/concept_intervention/qwen35_4b_k_diagnostic_v1/`.

PCA centering is used only to learn a reconstruction metric. Sparse pursuit always receives the original uncentered token directions. Every transformed solve recomputes transformed atom norms and saves raw-space reconstructions.

## Registered target and null families

Targets are exact shared logistic probes, train+validation-only class-mean differences, paired-template concept means at `assistant_boundary` and `concept_token_end`, and label-blind raw activations from unique held-out source texts. Raw text selection uses only text SHA-256 and token length.

Null families remain scientifically distinct:

- `iid_full_cardinality`: full best-atom search over a matched-norm iid dictionary.
- `iid_cardinality_sweep`: the same search at the five explicit fractions.
- `random_prefix_same_k`: exactly the first K predetermined random atoms; no vocabulary search.
- `orthogonal_rotation`: inverse target rotation in the active metric dimension.
- `label_permutation_probe`: group-safe, split-preserving, fixed-C conditional null with no test-row input.

The index reports first crossing minus one, consecutive-3 crossing minus one, right censoring, cumulative-excess peak, and fixed-budget reconstruction at K=4, 16, and 25. Fixed budgets are not occupancy estimates.

## Commands

Run from the dedicated run root so the pinned sibling three-method artifact paths resolve correctly:

```bash
export CODE_ROOT=/home/del6500/projects/J-lens
export RUN_ROOT=/data/del6500/J-lens/runs/qwen35_4b_k_diagnostic_v1
cd "$RUN_ROOT"
export PYTHONPATH="$CODE_ROOT/src"
PYTHON=/scr/del6500/J-lens/envs/three-method/bin/python

$PYTHON -m jlens_workspace.cli k-diagnostic validate \
  "$CODE_ROOT/Concept_intervention/configs/qwen35_4b_k_diagnostic_v1_pilot.yaml"
$PYTHON -m jlens_workspace.cli k-diagnostic prepare-targets CONFIG
$PYTHON -m jlens_workspace.cli k-diagnostic build-bases CONFIG
$PYTHON -m jlens_workspace.cli k-diagnostic run CONFIG --grid-index N
$PYTHON -m jlens_workspace.cli k-diagnostic index CONFIG
$PYTHON -m jlens_workspace.cli k-diagnostic report CONFIG
```

`validate` is read-only. `prepare-targets` loads the language model only for the registered concept-mean and raw-activation forward captures. `build-bases` and `run` load the tokenizer, cached unembedding/final RMSNorm, and exact lens without loading the full language model.

## Stage gates

1. Pilot: layers 11, 19, 27; `K_max=32`; 32 raw targets per layer; raw metric; all five null families.
2. Full raw: all seven concept layers plus raw layers 11, 19, 27; `K_max=64`; 128 raw targets per raw layer. It refuses to initialize until `pilot/index.json` is complete.
3. Transformed: layers 11, 19, 27; PCA and two regularized whitening metrics; no raw-activation targets; iid full, rotation, and prefix nulls. It refuses to initialize until `raw_metric_full/index.json` is complete.
4. Report: refuses to run until pilot, full raw, and transformed indexes are complete.

An existing root, stage, target, basis, or shard identity is reusable only when its saved identity and hashes match exactly. Partial or mismatched shards fail closed.

## Launchers

Local sequential/default launcher (set `K_DIAGNOSTIC_WORKERS` only when resources allow concurrent shard processes):

```bash
Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v1.sh pilot
Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v1.sh full
Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v1.sh transformed
Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v1.sh report
```

Slurm preparation, sealed array, and dependent index:

```bash
Concept_intervention/scripts/submit_qwen35_4b_k_diagnostic_v1.sh pilot
Concept_intervention/scripts/submit_qwen35_4b_k_diagnostic_v1.sh full
Concept_intervention/scripts/submit_qwen35_4b_k_diagnostic_v1.sh transformed
```

The submitter blocks only for target/basis preparation so it can read the sealed dynamic grid count; the scientific shards run as a bounded array.

## Offline verification

Stage 0 does not run the model or pilot:

```bash
PYTHONPATH=src /scr/del6500/J-lens/envs/three-method/bin/pytest -q tests/test_k_diagnostic.py tests/test_cli.py
/scr/del6500/J-lens/envs/three-method/bin/ruff check .
```

The diagnostic test module covers rotation equivalence, cardinality effects, nested random prefixes, transformed-coordinate formulas and raw reconstruction, group-safe fixed-C permutations without test inputs, target construction, exact-sphere theory, strict configs, unique scientific identities, append-only targets, and fail-closed shard resume/indexing.
