# J-space study configurations

These YAML files define direction-owned J-space studies: model and tokenizer
identity, fitted-lens compatibility, layers, matrix convention, chunking,
precision, and output paths. The reusable implementation lives under
`src/jlens_workspace/j_space/`.

Raw, centered, row-normalized, and RMSNorm-weighted matrices are distinct
scientific objects. Do not silently change an existing config to switch among
them or overwrite a prior output; copy the config, give the study and output a
new identity, and record the changed coordinate convention.
