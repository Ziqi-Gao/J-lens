# AxBench Concept500 `9b/l20` data audit

Audit date: 2026-07-13. This report covers the two Parquet files used by the
formal abstract-concept probing run, not the 96-row built-in smoke fixture.

## Identity and integrity

| Item | Value |
| --- | --- |
| Dataset | `pyvene/axbench-concept500` |
| Revision | `ad8a5d60c4616b599c24dd6689f05f696ec610f3` |
| Variant | `9b/l20` |
| License | `CC-BY-4.0` |
| Train file | 10,358,530 bytes; SHA-256 `5fb3042cc484a3e194f1043aab047bc1cfbe4efe884131d0113c75e9d2da599a` |
| Test file | 9,914,517 bytes; SHA-256 `090ff8292a331aab3776a836187406e49d8614864b250d9a1769f537da438f42` |

Both files contain one Parquet row group. The preparation script verifies both
hashes before writing any derived record.

## Source structure and quality

The train file has 36,216 rows: 36,000 positives (72 for each of 500 concept
IDs) and 216 shared generic negatives using the upstream `EEEEE`/`-1`
sentinel. The test file has 38,023 rows: 18,000 positives, 18,000 ordinary
negatives, and 2,023 polysemantic hard negatives. It supplies 36 positive and
36 ordinary negative rows for every concept.

The common fields are `input`, `output`, `output_concept`, `concept_genre`,
`category`, `dataset_category`, and integer `concept_id`; test additionally has
`sae_link` and `sae_id`. There are no null values. Every row has
`dataset_category=instruction`. Train genres contain 24,120 text, 9,288 code,
and 2,808 math rows; test contains 26,156 text, 7,844 code, and 4,023 math rows.

One exact duplicate exists in the full train file and none in the full test
file. The seven selected concepts and the 216-row generic-negative pool contain
no empty prompt/response and no exact within-concept duplicates. There is no
exact selected train/test overlap.

Whole-source word counts are:

| Source | Field | Median | P95 | P99 | Maximum |
| --- | --- | ---: | ---: | ---: | ---: |
| train | input | 10 | 41 | 66 | 102 |
| train | output | 74 | 110 | 115 | 150 |
| test | input | 10 | 45 | 70 | 97 |
| test | output | 64 | 108 | 113 | 128 |

## Selected abstract concepts

| Remote ID | Concept | Train positive | Test positive | Test ordinary negative | Test hard negative |
| ---: | --- | ---: | ---: | ---: | ---: |
| 68 | uses of deception or pretense | 72 | 36 | 36 | 3 |
| 82 | expressions of commitment to collaboration and community responsibility | 72 | 36 | 36 | 2 |
| 114 | references to power dynamics and political figures | 72 | 36 | 36 | 2 |
| 178 | questions and statements challenging beliefs or assumptions | 72 | 36 | 36 | 1 |
| 195 | expressions related to admiration and respect for individuals | 72 | 36 | 36 | 3 |
| 212 | expressions of vulnerability and honesty about personal struggles | 72 | 36 | 36 | 2 |
| 361 | concepts and discussions related to creativity and innovation | 72 | 36 | 36 | 2 |

The allowlist fixes both numeric IDs and exact upstream concept strings. This
prevents a later dataset revision from silently changing concept identity.

## Derived probing dataset

For each concept, all 72 training positives are retained. Seventy-two generic
training negatives are selected deterministically from the official shared
pool using the seed and concept ID. Each label is split into 54 train and 18
validation records. Official test rows are never resplit: all 36 positives, 36
ordinary negatives, and available hard negatives are retained.

The resulting artifact has fingerprint
`sha256:81d5079fafc53f4efa5ef64f7f0075058c36b52b8d023d10441420a9cf5bce0f`
and 1,527 records in 1,514 normalized-prompt groups:

| Split | Negative | Positive | Total |
| --- | ---: | ---: | ---: |
| train | 378 | 378 | 756 |
| validation | 126 | 126 | 252 |
| test | 267 | 252 | 519 |

All responses sharing a normalized prompt within a concept are assigned to one
atomic group. A two-dimensional exact allocation keeps these groups intact
while preserving the 54/18 per-label train/validation counts; no selected
normalized prompt crosses train/validation or the official train/test boundary.

Each binary probe therefore fits on 54 positive and 54 negative train rows,
tunes regularization with grouped five-fold CV, inspects 18+18 validation rows,
refits on 72+72 train/validation rows, and evaluates once on 73–75 official
test rows. The 504 negative train/validation records across seven tasks use 204
unique generic examples; reuse occurs only across separate concept IDs, never
within one binary task. Exact content is unique within each concept across all
splits.

Prepared prompt/response word counts have medians of 9/83, P95 values of
48/125, and maxima of 95/150. A 512-token activation cap should cover most
examples, but the GPU run records truncation through the fixed tokenizer and
must be checked before interpreting probe failures.

## Suitability and limitations

This dataset is large enough for a regularized first-pass linear probe and is a
substantial improvement over the 96-row smoke fixture. It is still not evidence
that a direction is causally steerable: the source contains synthetic
concept-conditioned responses, only 72 positive training examples per concept,
and a shared generic-negative distribution. Claims should therefore require
held-out AUC/balanced accuracy, layer and seed stability, lexical controls,
hard-negative performance, and later intervention dose-response comparisons.

The prepared JSONL and source Parquet files live under ignored `artifacts/` and
`.cache/` paths. Recreate them with
`Concept_intervention/scripts/prepare_axbench_concept500.py`; the checked-in
allowlist is the small, reviewable provenance object.
