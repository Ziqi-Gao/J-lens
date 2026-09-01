# Compatibility configurations

These flat YAML files are preserved inputs for registered or historical
Concept Intervention runs. Their paths may appear in manifests, launchers, and
durable provenance, so reorganizing the Python package does not move or rename
them.

Do not add a new versioned experiment here. Create it below
`Concept_intervention/experiments/<family>/<design-version>/`, register its
`experiment.yaml`, and keep design-owned configs with that manifest. Change an
existing flat YAML only when its registered protocol explicitly permits the
change; an ad-hoc rerun must use a copied config and a new output directory.

Reusable validation and workflow logic belongs in `src/jlens_workspace/`, not
in YAML-specific launcher code.
