# Abstract-concept data

The concept-steering direction uses one strict JSONL contract for local data and
adapted external data. The bundled smoke set is
`Concept_intervention/data/builtin_abstract_concepts.jsonl`.

## Record contract

Every line contains exactly these fields:

| Field | Meaning |
| --- | --- |
| `concept_id` | Stable, namespaced identifier. |
| `concept_name` | Human-readable concept name. |
| `definition` | Operational definition held constant for that ID. |
| `abstractness` | Must be the exact string `abstract`. |
| `label` | Integer `1` for presence and `0` for absence; booleans are rejected. |
| `text` | Non-empty text to encode or classify. |
| `prompt`, `response` | Optional generation context fields. They are empty for text-classification data and never construct the probe representation. |
| `split` | `train`, `validation`, or `test`. |
| `group_id` | Global semantic-source identifier. A group may contain both labels, but must stay within one concept and one split. |
| `source` | Dataset/provenance identifier. |
| `license` | License applying to the record. |

`ConceptExample.from_dict` rejects missing and unknown keys. Dataset validation
also rejects inconsistent concept metadata, reused normalized content within a
concept, group leakage, missing concepts/splits/labels, and counts below a
requested per-label and per-concept minimum. A negative may be reused for different
concepts because duplicate detection is concept-scoped.

```python
from jlens_workspace.data import load_jsonl

examples = load_jsonl(
    "Concept_intervention/data/builtin_abstract_concepts.jsonl",
    min_per_label_per_split=2,
)
```

The human-written English smoke set contains 96 examples over six concepts:
honesty, justice, compassion, uncertainty, creativity, and power. Each concept
has eight positive and eight negative examples. For each label, four are in
train, two in validation, and two in test. The 96 records form 48 matched
groups: every group contains one positive and one negative response to the same
label-neutral prompt, under the same concept and split. This pairing removes an
easy prompt-wording shortcut and keeps counterfactual variants atomic during
resplitting. The set is suitable for fast schema, hook, and end-to-end smoke
tests; it is too small to support substantive scientific claims.

## Deterministic preparation

`deterministic_group_split` hashes
`(seed, concept_id, label-composition, group_id)`, where label composition is
the group's `(#negative, #positive)` count tuple. It allocates whole groups
within each concept/composition stratum, so input row order does not affect the
assignment. A group may deliberately contain mixed labels, as the paired smoke
groups do, but it may not mix concepts and it is never divided across splits.
There must be at least one independent group per active split for every
stratum; otherwise preparation fails instead of silently creating a split with
missing coverage.

```python
from jlens_workspace.data import load_jsonl, prepare_examples

raw = load_jsonl(
    "Concept_intervention/data/builtin_abstract_concepts.jsonl",
    validate=False,
)
prepared = prepare_examples(
    raw,
    output_dir="artifacts/data/smoke_seed_42",
    seed=42,
    ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
)
print(prepared.statistics.to_dict())
print(prepared.fingerprint)
```

Preparation writes `train.jsonl`, `validation.jsonl`, `test.jsonl`,
`statistics.json`, and `manifest.json`. The fingerprint is a SHA-256 digest of
canonical records and is independent of row order. Directory loading obeys the
three files declared by `manifest.json`; auxiliary files such as J-lens fitting
prompts cannot be mistaken for probe examples.

## Formal GoEmotions binary tasks

The formal Qwen experiment uses the human-annotated
[`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions)
`simplified` split at revision
`add492243ff905527e67aeb8b80c082af02207c3`. The source dataset is described in
the [GoEmotions paper](https://aclanthology.org/2020.acl-main.372/). The
checked-in allowlist pins the SHA-256 of all three Parquet files.

The seven initial abstract social/affective concepts are admiration, approval,
curiosity, disapproval, gratitude, love, and optimism. The formal task is full
one-vs-rest: a row is positive when the target label occurs in its GoEmotions
label list and negative otherwise. Multilabel rows are retained. There is no
class-balancing subsample; `class_weight="balanced"` is applied by the probe.

The pinned files contain 54,263 raw rows. Exact normalized duplicates within a
split are canonicalized (232 rows removed), and every row belonging to a text
seen in more than one official split is excluded (170 rows across 82 normalized
texts). This leaves 53,861 unique, split-safe source texts: 43,112 train, 5,370
validation, and 5,379 test. Every concept therefore has 53,861 binary examples.
Natural positive totals are 5,093 admiration, 3,673 approval, 2,712 curiosity,
2,577 disapproval, 3,271 gratitude, 2,523 love, and 1,967 optimism.

The audit JSONL expands those sources across seven binary tasks (377,027 rows)
so every label is independently reviewable. Before model inference, capture
collapses the global `group_id` back to 53,861 source rows and writes one
`[53861,7]` label matrix with zero missing entries. Thus each text is forwarded
through Qwen exactly once per activation capture, not seven times.

The preparation script also selects 1,000 label-blind training texts of at
least 24 whitespace-delimited words for J-lens fitting. These prompts may
overlap the probe corpus: J-lens fitting uses no concept labels and overlap is
not a leakage path. This prevents discarding valid probe rows merely to fit the
model-wide Jacobian map.

```bash
.venv/bin/python scripts/prepare_go_emotions.py \
  --allowlist Concept_intervention/data/go_emotions_7concept_full_allowlist.json \
  --train artifacts/source_data/go_emotions_simplified_add492243ff905527e67aeb8b80c082af02207c3/train.parquet \
  --validation artifacts/source_data/go_emotions_simplified_add492243ff905527e67aeb8b80c082af02207c3/validation.parquet \
  --test artifacts/source_data/go_emotions_simplified_add492243ff905527e67aeb8b80c082af02207c3/test.parquet \
  --output artifacts/data/go_emotions_7concept_full_ovr_v1 --seed 42
```

The pinned full allowlist emits 3,000 deterministically ordered fit prompts.
Production configs use 1,000 at a time. Set `lens.fit_prompt_offset` to `0`,
`1000`, or `2000` and use a distinct lens/output path to run three disjoint
prompt-sample replications without changing the probe examples.

The previous fixed 1,000-positive/1,000-negative, single-label dataset remains
available only as a balanced control and is not the formal probing corpus.

The probed representation is always the last non-padding token of `text` at
`resid_post`. There is no separate dataset vocabulary: the pinned Qwen
tokenizer encodes these texts, and J-direction token IDs are decoded by that
same tokenizer. Qwen's padded output-head rows beyond `len(tokenizer)` are
excluded with a zero-copy row view, so every searched row has a decodable token
ID.

## AxBench adapter

The optional adapter targets
[`pyvene/axbench-concept16k`](https://huggingface.co/datasets/pyvene/axbench-concept16k)
at a pinned revision and one explicit model/layer parquet (`2b/l20` or
`9b/l20`). It supports both Hugging Face streaming and ordinary loading. The
verified upstream fields are `input`, `output`, `output_concept`,
`concept_genre`, `category`, `dataset_category`, and `concept_id`. Positive rows
carry a concept name and nonnegative ID; generic negative rows use the upstream
sentinels `output_concept="EEEEE"` and `concept_id=-1`. The adapter assigns a
disjoint slice of those generic negatives to each selected concept and records
the discovered positive ID in a namespaced local ID.

The checked-in allowlist at
`Concept_intervention/data/axbench_abstract_allowlist.json` intentionally has
`remote_output_concept=null` and `verified_against_revision=null`. The six
human concepts are research targets, not claims about exact GemmaScope strings.
Consequently `prepare_axbench` fails **before importing `datasets` or making a
network request**. This is deliberate: a local word such as `honesty` must not
be converted into an invented remote concept ID.

To enable a concept:

1. Load the pinned source with `load_axbench_source` and call
   `discover_axbench_concepts(rows, search_terms)` to obtain candidate positive
   `output_concept` strings and their observed IDs. This read-only discovery
   path permits an unverified allowlist; `prepare_axbench` does not.
2. Inspect candidate examples and decide whether the GemmaScope description is
   the intended abstract concept.
3. Copy the exact `output_concept` string into the allowlist and set
   `verified_against_revision` to the allowlist's pinned revision.
4. Run preparation. The adapter scans both an iterable dataset and an
   `IterableDataset`, checks the observed ID, requires enough positives and
   negatives, and fails if the source schema or identity differs.

```python
from jlens_workspace.data import prepare_axbench

prepared = prepare_axbench(
    "Concept_intervention/data/axbench_abstract_allowlist.json",
    "artifacts/data/axbench_abstract",
    streaming=True,
    per_label=8,
    seed=42,
)
```

Ordinary loading uses the same call with `streaming=False` and may download a
large parquet file. `datasets` is optional and is not required for local smoke
data or unit tests.

## Legacy Concept500 baseline

An earlier baseline used the pinned `9b/l20` train/test files from
`pyvene/axbench-concept500`, not the Concept16K discovery adapter above. The
reviewable allowlist is
`Concept_intervention/data/axbench_concept500_abstract_allowlist.json`; the
complete audit is in `docs/axbench_concept500_9b_l20_eda.md`.

Recreate the ignored prepared artifact with:

```bash
.venv/bin/python Concept_intervention/scripts/prepare_axbench_concept500.py \
  --allowlist Concept_intervention/data/axbench_concept500_abstract_allowlist.json \
  --train .cache/axbench-concept500/9b/l20/train.parquet \
  --test .cache/axbench-concept500/9b/l20/test.parquet \
  --output artifacts/data/axbench_concept500_abstract_v1 \
  --seed 42 --validation-fraction 0.25
```

For every selected concept, preparation keeps all 72 positive training rows,
deterministically selects 72 official generic negatives, and assigns complete
normalized-prompt groups to 54/18 train/validation rows per label. The official
36-positive/36-negative test rows and all available hard negatives remain in
test. File hashes and the derived fingerprint are verified before model work.

Concept500 positives often contain explicit target words, and training
negatives come from a shared generic pool. A high probe AUC is therefore a
feature-detection baseline, not yet evidence of abstract or causal semantics.
Before a steering claim, add TF-IDF/keyword baselines, keyword masking,
cross-dataset transfer, and intervention controls.
