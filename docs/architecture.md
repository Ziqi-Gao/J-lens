# Repository architecture

J-lens has two sibling scientific programs and one shared implementation
package. The sibling directories are not Python source mirrors: they are the
authoritative experiment surfaces that define what is run and how the result is
interpreted. `src/jlens_workspace/` contains the reusable implementation of
those definitions.

## Ownership planes

| Plane | Owns | Must not own |
| --- | --- | --- |
| `Concept_intervention/` | registered designs, protocol metadata, small input manifests, compatibility configs and launchers, scientific reports | reusable algorithms, environments, generated tensor artifacts |
| `J_space/` | matrix-study configs, direction-specific launchers, design notes, scientific reports | Concept Intervention inputs or outputs, reusable algorithms |
| `src/jlens_workspace/` | typed Python implementations, schemas, numerical kernels, workflow APIs, scheduler adapter | experiment identity, mutable runtime state, large artifacts |
| `/data/del6500/J-lens` | durable datasets and immutable scientific artifacts | source code, locks, caches |
| `/scr/del6500/J-lens` | environments, caches, logs, locks, scheduler state, temporary checkpoints | scientific completion evidence by itself |

The direction directories answer **what scientific experiment is being run**.
The package answers **how that experiment is computed**. Moving reusable code
under `src/` must never move or erase experiment identity, design YAML, input
manifests, protocols, or reports from the direction that owns them.

## Python responsibility map

```text
jlens_workspace/
|-- foundation/                 model, config, activations, artifacts, Jacobian
|-- numerics/                   direction-neutral numerical primitives only
|-- data/                       shared preparation and validation interfaces
|-- concept_intervention/
|   |-- protocol/               shared rows, controls, strengths, artifact schemas
|   |-- probing/                correlational concept measurement
|   |-- geometry/               alignment, sparse pursuit, K diagnostics
|   |-- steering/               J-component, ITI, and RAPTOR interventions
|   |-- evaluation/             method-neutral evaluation over frozen outputs
|   `-- reporting/              artifact-only report generation
|-- j_space/                    A_l operators, spectra, bases, and workflows
|-- scheduler/                  ServerScheduler v2 validation and execution
`-- cli.py                      stable user-facing command surface
```

Compatibility modules at historical import paths may re-export canonical
objects during migration. New implementation code must use the canonical
paths. A compatibility module must contain no new scientific logic.

## Scientific dependency rules

The Concept Intervention stages are distinct even when one consumes a frozen
artifact produced by another:

```text
data + protocol -> probing -> geometry -> frozen direction/geometry artifacts
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
              J-component                  ITI                    RAPTOR
                    +------------------------+------------------------+
                                             |
                              matched, signed generation outputs
                                             |
                              evaluation -> artifact-only reporting
```

- Probing may not import steering or evaluation.
- Steering consumes versioned probe/geometry artifacts through protocol
  schemas; it does not refit the shared probe implicitly. ITI's method-internal
  head probe remains owned by `steering/iti/`.
- J-component, ITI, and RAPTOR may share protocol and evaluation interfaces but
  may not import one another's implementation.
- Evaluation reads common output schemas and immutable artifacts, not steering
  implementations.
- Reporting must not load a model, fit a probe, intervene, or mutate source
  artifacts.
- `j_space` depends only on foundation and direction-neutral numerics. It may
  not import Concept Intervention scripts, modules, or artifacts.

The causal evidence gate is unchanged: signed strengths and matched full, J,
non-J, and random controls are required before a causal steering claim. Probe
AUC, token alignment, or geometry alone remain correlational evidence.

## Execution and scheduling boundary

The scheduler package is an operational adapter, not a scientific stage. It
validates a stored ServerScheduler v2 job, selected execution profile,
allocation, and environment; dispatches a fixed allowlisted command; and emits
a completion receipt only after the task-specific scientific artifact contract
passes. Process exit zero is never sufficient completion evidence.

The current catalog distinguishes `sealed_source`, `derivation`, and
`read_only` output policies. Completed registered source roots are never
reopened: sealed tasks fail before launch. Derived revisions are typed by the
experiment registry, generated in sibling attempt-staging directories, expose
their final root only by atomic publish after `index.json`, and may be adopted
after a lost scheduler receipt only when the entire hash chain revalidates.
Operational retry remains `attempt-NNN`; failure is never a reason to invent a
new scientific design or derived revision.

The J-lens task/profile catalog is the project-side source of truth for the
central registration proposal. Installing or enabling that proposal and
managing the central service belong to a separately owned ServerScheduler
maintenance task. The legacy local scheduler remains available until the
central CPU, cross-NUMA, GPU, completion-receipt, and dependency-chain cutover
gates have all passed.

Scheduler transport must use new scheduler-owned entrypoints. A path registered
as a historical protocol, config, or launcher is an immutable compatibility
input and must not be repurposed to admit ServerScheduler or redirected to a
new derived output. The scheduler-only task wrappers may delegate an unchanged
read-only historical task body, but all new derivation routing remains outside
the registered launcher surface.

`scheduler/dag.py` is the side-effect-free source of truth for dependency
release. It expands the three-method smoke/full/finalize/rescore graph and the
ordered K-diagnostic stages and waves, but it never creates or submits a
request. Publishing those validated invocations remains a separately approved
central operation.

## Migration policy

1. Move one responsibility at a time and retain a tested compatibility import.
2. Preserve registered experiment paths, output paths, schemas, coordinates,
   model pins, and artifact hashes.
3. Add an import-boundary test before removing a compatibility path.
4. Run focused numerical equivalence tests and the ordinary offline suite at
   each boundary.
5. Remove a deprecated launcher or local-scheduler component only in a later,
   explicit cleanup after no service or registered experiment references it.

This migration changes code ownership and discoverability. It does not create a
new scientific design, rerun an experiment, or reinterpret existing evidence.
