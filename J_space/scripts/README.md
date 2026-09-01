# J-space launchers

These files are thin direction-owned entrypoints for J-space configurations.
They validate inputs and call the packaged `jlens-workspace matrix` workflow;
operator, spectrum, rank, and subspace logic belongs under
`src/jlens_workspace/j_space/`.

- `smoke_offline.sh` is the ordinary offline validation path.
- `run_tiny_integration.sh` is an opt-in real-model dependency check, not a
  scientific experiment.
- `run_qwen35_4b.sh` and `run_qwen35_4b.slurm` are preserved Qwen transport
  entrypoints whose config can be overridden without editing the launcher.

J-space launchers may not import or consume Concept Intervention scripts or
artifacts.
