# Launcher inventory

This directory is a compatibility and transport surface. Scientific algorithms
belong in `src/jlens_workspace/`; new designs belong under
`Concept_intervention/experiments/` with a thin design-owned entrypoint when
one is necessary.

The existing launchers fall into these operational groups:

| Group | Naming examples | Status |
| --- | --- | --- |
| ServerScheduler v2 | `server_scheduler_entrypoint.sh` | canonical central adapter entrypoint |
| Registered Slurm transport | `run_*.slurm`, `submit_*.sh` | preserved compatibility for registered designs |
| Offline and tiny checks | `smoke_offline.sh`, `run_tiny_integration.sh` | supported validation entrypoints; tiny integration is opt-in |
| Legacy local DAG | `*local*`, `setup_three_method_local.sh`, `bootstrap_three_method_local.sh` | retained until central cutover gates pass |
| Shared environment helpers | `three_method_env.sh`, `three_method_submit_env.sh` | compatibility dependencies of existing launchers |

An existing path is not proof that it is the preferred entrypoint. Consult the
experiment registry first. Scripts must only validate/translate arguments and
invoke a packaged command; they must not acquire new reusable scientific logic.

The legacy local DAG is intentionally not deleted during the package
reorganization. Removal requires successful central CPU, cross-NUMA and GPU
pilots, a validated dependency chain, no live service references, and a
separate cleanup review.
