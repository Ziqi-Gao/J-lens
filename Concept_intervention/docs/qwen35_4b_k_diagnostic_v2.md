# Qwen3.5-4B K diagnostic v2 preregistration

This identity supersedes the unexecuted v1 design. It diagnoses why concept-target
reconstruction can cross the null at `K <= 4`; it does not overwrite prior occupancy
or intervention artifacts, perform generation, or use an LLM judge.

## Immutable inputs and target construction

The model, tokenizer, exact local lens, activation arrays, row/probe manifests, and
held-out JSONL are revision- or SHA-256-pinned in each v2 YAML. Every concept capture
saves the exact rendered prompt and SHA-256, token-ID hash, subject token span, and
assistant-boundary token index. The target index is append-only.

The old `anthropic_style_concept_mean` name is prohibited in v2. The estimand is
`seven_emotion_label_contrast`, with three separately reported arms:

- `label_explicit`: paired one-vs-six contrasts whose prompt includes the label.
- `label_free_definition`: target definitions/paraphrases that do not contain the
  target label; the prompt manifest and hash are saved.
- `four_unrelated_topics_control`: the same label-free target definitions contrasted
  with an immutable, hashed bank of exactly four non-emotion topics. It is a narrow
  four-topic control, not a broad unrelated-concept population.

`assistant_boundary`, `concept_token_end`, and `definition_span_end` are distinct
capture cells. They are never pooled into one family median. Statistical targets are
the shared logistic probe and train+validation class-mean difference. Raw activations
have a persisted `selection_rank` and explicit stage membership. A sealed pilot
membership manifest is byte-stable after the full store appends ranks 32--127.

## Scientific repeats versus physical execution

A logical replicate is one `(target, layer, metric, null family, seed, fraction)`.
A physical bundle is one `(target, layer, metric)` and contains every registered null
family/seed/fraction for that target. One process loads the tokenizer/lens/unembedding
once, computes atom norms once, and runs the real pursuit once. Logical shard
directories remain independent, hash-pinned, and auditable. They save their null
arrays plus `real_reference.json`; real curves, support, coefficients, and transformed
raw reconstructions are hash-pinned once per bundle. A bundle-level
`fixed_budget_real.json` stores, for K=1,2,4,8,16,25,32,64, the target/reconstruction
cosine, residual and normalized residual, selected token IDs and decoded tokens,
coefficients, and selected-atom Gram condition number. Transformed bundles also
store metric- and raw-space residual/explained values. Shards contain only the
verified bundle hash reference, never a copy of this payload. Bundle manifests assert
one real solve and one norm pass. Exclusive-create locks, immutable manifests,
complete markers, shard re-audit on resume, and exact cache hashes make partial,
concurrent, or mismatched reuse fail closed.

The v2 physical executor is pinned as `kdiag_null_batch_v1`; this marker is present
in bundle, shard, runtime, microbenchmark, resource-preflight, and stage-index
identities. It is not applied to v1 schema-2 artifacts, which retain their scalar
resume/index contract. For each IID seed, one ephemeral float64 provider materializes
the exact legacy Philox `key=[seed,chunk_index]` unit directions once. Every
cardinality cell uses a prefix view with its original independently selected
matched-norm subset, preserving the legacy chunk order (`unit @ residual`, then norm
scaling). Full-cardinality IID and sweep fraction 1 share one curve computation but
keep distinct logical shard identities and hash references. The 31 inverse-rotated
targets and the 31 applicable permutation targets each use one batched
same-dictionary pursuit, without changing seeds, target vectors, K, or null meaning.

The shared-direction provider is bundle/seed-local and never becomes a scientific
dictionary artifact. Its actual allocation gate records observed available RAM,
float64 storage, generation workspace, peak bytes, a 6 GiB execution-safety cap,
and an 8 GiB post-allocation reserve in the immutable bundle manifest. The 6 GiB cap
admits the production exact-float64 5,248,421,888-byte (4.888 GiB) provider while
leaving headroom inside the registered 12 GiB FSM `kdiag-bundle` host-RAM lease
(and the 64 GiB Quest template); it is not a scientific design choice or experiment
resource budget. Resume independently
rechecks current RAM while retaining the originally sealed gate evidence. The CLI
benchmark copies and validates that executed manifest plan; `resource-preflight`
compares stable plan fields and separately validates both copies' dynamic checks.
Old scalar benchmark/runtime approvals cannot authorize this execution version.
Each logical shard is assembled and fsynced in a sibling staging directory, verified,
and atomically renamed into the registered `shards/` root. An interrupted staging
directory is invisible to the grid; completed new-version shards and the single
`real_complete.json` payload remain reusable on retry.

Each `seven_emotion_label_contrast` target also owns an immutable float64 `[T,D]`
per-template contrast array. Its relative path, shape, canonical raw-float64 hash,
and file SHA-256 are part of the target index and resume identity. All three arms and
both registered capture positions are required, so later bootstrap or template-level
diagnostics never require another model forward pass.

The registered upper counts are:

| stage | physical bundles/jobs | logical replicates | disk/inodes |
|---|---:|---:|---|
| pilot | 264 | 66,774 | measured from stage benchmark |
| full raw | 776 | 195,486 | measured from stage benchmark |
| transformed | 1,176--1,344 | 109,368--124,992 | measured from stage benchmark |
| total | 2,216--2,384 | 371,628--387,252 | measured stage plus reserved hard budgets |

The transformed range arises only when data-derived `j_pca_k90` is identical to a
fixed registered rank and is de-duplicated. Static creation stops above the YAML
limits for bundle count or logical count. Slurm arrays operate on bundles and
are split into registered chunks of at most 900, so no 66k--195k logical array is
submitted and cluster `MaxArraySize` is respected.

## Stage microbenchmarks and hard resource gate

Before any ordinary bundle in each stage, the launcher runs that stage's largest
`K_max * metric_dimension * (1 + logical_shards)` cost proxy. Transformed atom-norm
and matrix multiplication costs therefore receive their own representative benchmark.
The immutable benchmark records wall time and measured bundle/shard bytes and inodes.
No benchmark value is fabricated in the repository; it must be measured on exact
pinned artifacts under the registered execution backend. Every v2 microbenchmark and
ordinary bundle runs a fail-fast scheduled-single-CUDA gate before loading the target,
model, or dictionary: either a live FSM `JLENS_LOCAL_LEASE_ID` or a Slurm allocation
must exist, CUDA must be available, and exactly one logical device at index 0 must be
visible. GPU model names are discovered, never allow-listed. Microbenchmark schema v5
records the stable backend class (`fsm_local_scheduler` or `slurm`), visible GPU
name/count, PyTorch version, and CUDA version plus a canonical SHA-256; hostname and
the random lease ID are deliberately excluded from cross-bundle identity. Resource
preflight and ordinary bundle approval reject a missing allocation or malformed/
hash-mismatched record. An ordinary bundle also requires its live backend/hardware
payload to equal the approved benchmark exactly before recomputation, and every
runtime plus the final stage index re-audits that identity.

The conservative preflight formulas are:

- `stage_gpu_hours = benchmark_wall_seconds * physical_bundle_count / 3600`;
- `stage_bytes = ceil(1.25 * (physical_count * measured_bundle_bytes +
  logical_count * measured_max_shard_bytes + rotation_cache_bytes))`;
- inode projection uses the same formula with measured inode counts;
- the full-experiment bound uses the measured current stage and reserves unmeasured
  stages at their registered hard budgets.

Before a full exact-Haar build, `rotation-cache-preflight` measures one representative
Gaussian-QR construction, records bytes per unique `(algorithm, dimension, seed)`
entry and maximum dimension, and projects wall time using the conservative
`(D_sample/D)^3` QR scaling with a 1.25 factor. It also projects total cache bytes and
the number of metric/transform binding rows. A projection over 86,400 seconds is
rejected before any full cache build. The effective cache disk gate is
`min(max_rotation_cache_gib, current_stage_disk_cap, full_experiment_disk_cap minus
the registered reserves for the other stages)`, namely no more than 25/75/75 GiB for
pilot/full/transformed under the registered v2 budgets; the analogous stage/full
inode limits are also enforced before the full build. The immutable preflight pins
the complete configuration SHA-256 and binding-plan SHA-256. Both identities and
the preflight file hash are required by the cache index and ordinary resource
preflight, so changing a cache budget invalidates an older approval.

Each bundle must take at most 7,200 seconds. Stage GPU-hour budgets are 500, 2,500,
and 3,000; disk budgets are 25/75/75 GiB; inode budgets are 0.5/2/2 million. The
full-experiment caps are 6,000 GPU-hours, 175 GiB, and 4.5 million inodes.
`resource-preflight` must pass before the first local-scheduler or Quest wave and
after every wave.
Any observed overrun, failed hash audit, partial cache, stale benchmark, or resource
budget excess stops later submission/execution.

## Nulls and resolution

The raw stages retain five scientifically distinct nulls: full-cardinality matched-
norm iid search, five fixed cardinality fractions, fixed nested random prefix,
orthogonal rotation, and label permutation. Transformed metrics retain exactly
full-cardinality iid, fixed nested random prefix, and orthogonal rotation; label
permutation is a raw statistical-target diagnostic and is not part of transformed
inference. Every iid/fraction, rotation, and applicable permutation cell has 31
registered seeds, so a direct empirical tail has resolution `1/(31+1)=0.03125`
instead of v1's unusable 1/4 or 1/6. The primary mechanism inference is not a null-
seed pseudo-sample: it uses target-level hierarchical bootstrap; null seeds calibrate
each target curve.

Orthogonal rotation remains exact Gaussian-QR Haar. A matrix is constructed once for
each `(algorithm version, dimension, seed)`, atomically saved, SHA-256 pinned, and
mmap-loaded by all target bundles. A separate immutable binding index pins every
metric/transform use to that shared matrix. Raw pilot/full share the same 31 matrices;
transformed metric spaces with the same dimension also share them. Each stage binding
index carries the current configuration and binding-plan identity and must exactly
cover the transform SHA-256, dimension, metric, layers, and seeds rebuilt from the
current target/transform artifacts. Every indexed matrix hash is compared with the
loaded immutable manifest, and index-stage shard audit forbids a fallback lookup for
keys missing from that audited mapping. Partial, locked, extra, stale-transform, or
hash-mismatched cache state fails closed. The cache-build preflight, not a hand-
estimated matrix count, records the exact unique-matrix and binding counts.

Group-safe permutation is performed separately in train and validation and never
accepts test labels. Logistic targets refit at the real fixed C. Class-mean targets
also permute labels and recompute the mean-difference direction. Each split records
Hamming distance, non-identity rate, effective permutable groups, and size strata.
Absence of any mixed-label equal-size stratum, or an identity draw, is an error and
not a valid null.

Rotation and permutation summaries persist production empirical diagnostics rather
than merely exposing a helper: every K has a marginal-gain upper-tail percentile and
p-value; fixed K=4, each fixed-budget reconstruction benefit, and cumulative benefit
have target-level empirical percentile/p-values. Every row records `B=31` and
resolution `1/32`, and report validation rejects missing or malformed diagnostics.

## Preregistered inference

- Primary target-level estimand:
  `Delta4(t) = median_b[E_null(t,b,K=4)] - E_real(t,K=4)`, where `E` is normalized
  residual squared error. Mechanism effects are registered numerator-minus-
  denominator contrasts of `Delta4`; rotation and label permutation use `Delta4`
  directly. `max_cumulative_excess` is adaptive over K and exploratory only; it
  cannot support a mechanism conclusion. `K_first` and consecutive-3 K are auxiliary.
- Positive comparison direction; minimum supported effect `0.01`; equivalence margin
  `0.0025`. One absolute target-energy percentage point at K=4 is the smallest
  preregistered practically interpretable shift; one quarter point is the equivalence
  margin. For cardinality slope, the unit is per log2 dictionary-fraction increase.
- Primary transformed checks: `j_pca_r256` and whitening floor `0.10`.
- 4,000 hierarchical bootstrap draws, seed 8844, 95% percentile CI. The hierarchy is
  layer, concept/source, then registered target/template-source cell. Null seeds are
  collapsed inside a target cell and are never resampled as independent targets.
- Holm--Bonferroni is applied jointly to the registered mechanism rows.
- `Supported`: CI lower bound is at least 0.01 and Holm-adjusted one-sided bootstrap
  p is at most 0.05. `Not supported`: CI upper bound is at most 0.0025.
  Otherwise: `Inconclusive`.

The report refuses to run unless all three indexes are complete and every summary
cell exactly matches the sealed target membership, metric/family/fraction grid, seed
identities, and minimum repeat counts. Every mechanism row contains effect, CI,
thresholds, numerator, denominator, pair key/count, cluster counts, multiplicity
result, conclusion, and resampling hierarchy.

| registered mechanism | contrast and pairing |
|---|---|
| target construction | logistic minus raw (nonpaired within common layer); logistic minus class mean (concept x layer paired) |
| lexical label and capture position | explicit minus label-free only at the shared assistant-boundary cell; explicit concept-end minus its assistant boundary and label-free definition-end minus its assistant boundary are separate within-arm concept x layer pairs; unlike end positions are never paired |
| baseline scope | four-topic control minus label-free one-vs-six, separately at assistant boundary and definition end (paired) |
| random search | random-prefix separation minus full-iid separation (target x layer paired); negative cardinality slope plus monotonic-target fraction |
| dictionary alignment | real minus Haar-rotated at the same target/layer |
| label structure | real minus group-safe permuted, separately for logistic and class mean |
| metric | PCA r256 minus raw and whitening 0.10 minus raw (target x layer paired) |

Right-censored `K=None` values are never silently removed to claim a population
median. Tables report only the observed-crossing distribution plus censoring count
and fraction. Figures 1 and 2 show censored points explicitly. Figure 4 first takes
the pointwise median across all 31 iid seeds within target, then summarizes across
de-duplicated targets. Real support is read once through the bundle hash reference;
no `first 500 shard` or repeated real-target weighting is allowed.

The theoretical curve is a one-sided signed-cosine isotropic reference.
`sqrt(explained@1)` equals maximum positive cosine only for the registered positive-
direction best single-atom step.

Support overlap is computed at K=4,16,25 for three non-pooled comparison classes:
logistic versus class mean; logistic versus each concept arm/capture position; and
class mean versus each concept arm/capture position. Rows save the pair identity,
target-vector cosine, and all three Jaccards. Figures 8 and 9 use registered `Delta4`
and paired transformed-minus-raw `Delta4` as their primary y values; adaptive
`max_cumulative_excess` appears only in explicitly exploratory output. The CLI report
prints the registered A--J numeric summary, with observed crossings and censoring
fractions instead of a misleading population median. Terminal item A restricts both
raw activation and logistic probe to their preregistered common layers 11, 19, and
27; logistic-only layers remain visible in aggregate and other applicable cells but
cannot enter that comparison.

## Resource-budget amendment after the first FSM benchmark

The first scheduler-managed FSM worst-bundle microbenchmark, run from commit
`3b39210`, completed before any ordinary pilot bundle was authorized. Its unchanged
1.25x projection measured 587,398 pilot inodes and a 4,587,398-inode full-experiment
upper bound. The original 500,000/4,500,000 caps therefore triggered the registered
fail-closed resource gate, while every GPU-hour, disk, runtime, hardware, and shared-
direction-memory check passed.

Before any ordinary bundle was run, the pilot inode cap was amended to 600,000 and
the shared full-experiment inode cap in all three v2 stage configurations was amended
to 4,600,000. This amendment changes only execution resource ceilings. The 1.25x
projection formula, artifact file schema, targets, null families and seeds, pursuit
solver, estimands, inference rules, and all logical/physical grid identities remain
unchanged. The rejected benchmark is not reusable under the amended configuration
hash and must remain recoverably quarantined.

## Commands

FSM local execution is scheduler-only. Inspect the immutable DAG, then run only the
pilot microbenchmark gate first:

```bash
Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v2.sh plan
Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v2.sh microbenchmark
```

After reviewing its scheduler selection, hardware hash, runtime, and approved
resource projection, the remaining stage commands are:

```bash
Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v2.sh pilot
Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v2.sh full
Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v2.sh transformed
Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v2.sh report
```

The top-level DAG fixes `RUN_ROOT` to
`/data/del6500/J-lens/runs/qwen35_4b_k_diagnostic_v2`, shares the existing
`/scr/del6500/J-lens/runtime/three-method/scheduler` lease namespace, and requires a
clean commit. CPU controls use host leases; target/basis construction uses an
exclusive `kdiag-targets` GPU lease; each physical bundle uses the shared,
capacity-gated `kdiag-bundle` profile. `xargs` launches only scheduler workers and
never launches Python/CUDA directly.

The retained Quest Slurm files are cluster templates. They request an A100 from the
Quest scheduler, but A100 is not a v2 scientific or artifact acceptance gate. They
prepare/cache exact Haar matrices, benchmark/preflight the stage, submit at most
900-element waves, preflight between waves, then index:

```bash
Concept_intervention/scripts/submit_qwen35_4b_k_diagnostic_v2.sh pilot
Concept_intervention/scripts/submit_qwen35_4b_k_diagnostic_v2.sh full
Concept_intervention/scripts/submit_qwen35_4b_k_diagnostic_v2.sh transformed
```

Do not submit a stage array until its representative benchmark and
`resource_preflight.json` have both been reviewed and have `approved=true`.

## Remaining design limits

This is a reconstruction diagnostic, not a causal intervention or behavioral test.
The iid sphere is a reference rather than a model of anisotropic token geometry.
Concept directions that average a registered prompt manifest treat the within-
manifest prompts as a fixed construction; the inferential observations are target
cells, not post-hoc prompt draws. The fixed-C logistic permutation is conditional on
the original regularization choice. Results cover one model revision, one lens, the
registered layers, and seven GoEmotions labels.
