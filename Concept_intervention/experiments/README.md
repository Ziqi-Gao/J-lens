# Experiment registry and versioning

This directory is the canonical index for versioned Concept Intervention
experiments.  It organizes experiment identity without relocating or rewriting
sealed configs, launchers, or artifacts from earlier runs.

## Four independent identities

Use exactly one namespace for each type of change:

| Identity | Form | Create it when |
| --- | --- | --- |
| Scientific design | `design-vN` | the hypothesis, data/split, method, selection rule, estimand, primary metric, or judge design changes |
| Protocol revision | `protocol-vN` | a compatible prompt, rubric, wire format, or schema changes without changing the estimand |
| Operational attempt | `attempt-NNN` | the same frozen design is resumed or retried after interruption, timeout, OOM, API failure, or parser repair |
| Derived revision | `revision-rN` | an immutable source artifact is rescored or reanalysed without regenerating its scientific samples |

Git commits, snapshot hashes, dates, and server names belong in manifests.  They
must not replace any of the four identities above.

In particular, a retry must never become a new `design-vN`.  Historical names
such as `llm_judge_pilot_v1` through `v16` are registered as legacy attempts of
one design and must not be copied as a naming pattern.

## Canonical layout

New experiments use this layout from their first commit:

```text
experiments/<family>/design-vN[-qualifier]/
├── experiment.yaml       registry manifest and code/artifact provenance
├── protocol.md           scientific protocol, or a registered protocol path
├── configs/              design-owned scientific YAML
├── steering/             steering overrides when the design compares interventions
├── derivations/
│   └── revision-rN.yaml  typed immutable-source/derived-output contract
└── launch_local.sh       one thin parameterized local entrypoint, when needed
```

Do not create new versioned YAML or copied launchers directly in
`Concept_intervention/configs/` or `Concept_intervention/scripts/`.  Those
directories are compatibility surfaces for already registered designs.  Shared
tested implementation remains in `src/jlens_workspace/`; this directory owns
only metadata, design-specific configuration, protocols, and thin entrypoints.

## Artifact layout

Existing artifact paths are immutable and remain where their provenance says
they are.  A future design should use:

```text
/data/del6500/J-lens/runs/<family>/<design-version>/
├── manifest.json
├── inputs/
├── stages/
├── evaluations/
├── attempts/attempt-NNN/
├── derivations/revision-rN/
└── final/index.json
```

Runtime state, locks, caches, and raw retry state belong under the matching
`/scr/del6500/J-lens` tree.  They are not durable completion evidence.

When an `experiment.yaml` declares `derivations`, every referenced revision
manifest is strictly validated. It binds the parent design and immutable source
completion markers, states that samples and estimand are unchanged, assigns a
J-lens-owned `/data` revision root, enumerates relative completion markers and
scheduler tasks, and requires `write_policy: write_once`. An interrupted run is
still an `attempt-NNN`; it must never be repaired by inventing a new
`revision-rN`.

## Required workflow

1. Search `registry.yaml` before creating a design.
2. Copy the `_template/experiment.yaml` manifest into the new design directory.
3. State what scientific change justifies a new `design-vN`; otherwise use a
   protocol, attempt, or revision identity.
4. Add the manifest to `registry.yaml` in the same commit as the new design.
5. Validate from the repository root:

   ```bash
   .venv/bin/jlens-workspace experiments validate
   ```

6. On this managed server, also check registered durable paths when auditing
   existing runs:

   ```bash
   .venv/bin/jlens-workspace experiments validate --check-artifacts
   ```

The second command is server-specific: it requires every registered path and
checks that every completion marker contains `"complete": true`. It is not
required in ordinary offline CI. Neither command modifies scientific artifacts.
