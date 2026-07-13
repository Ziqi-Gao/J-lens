# Research design and invariants

## Repository boundary

`Concept_intervention/` and `J_space/` are sibling experiment surfaces. They may
share stable implementation code from `src/jlens_workspace/`, but one direction
must not import scripts or outputs from the other.

## Coordinate conventions

- Activations are captured at the transformer block output (`resid_post`), which
  is the coordinate system used by the official Jacobian-lens estimator.
- Concept probes use the final non-padding token of the labeled text. Pooling
  modes are intentionally absent from the production configuration and API.
- A saved lens is valid only for the recorded model checkpoint, tokenizer,
  residual location, layer, BOS policy, and weight coordinates.
- `raw` means `A_l = W_U J_l`.
- `rmsnorm_weighted` means the linear frame
  `A_l = W_U diag(gamma) J_l`. The input-dependent RMS scalar is not folded into
  this matrix. Unsupported normalization layers fail explicitly.
- Probe coefficients are always exported in the original residual coordinates,
  even when feature standardization is used during training.

## Statistical gates

- Dataset groups never cross train/validation/test boundaries.
- Logistic-probe penalty strength is selected using training-only grouped or
  stratified cross-validation. The held-out test split is evaluated once.
- Every concept must have both labels in every required split; the production
  config additionally enforces at least 2,000 total examples per concept.
- Alignment claims report a random matched-norm control and stability across
  seeds or bootstrap samples.
- A PCA basis is called "minimal" only relative to an explicit cumulative-energy
  threshold. Exact matrix rank and effective rank are reported separately.

## Artifact contract

Every expensive run writes a manifest with exact model/tokenizer/lens/data
revisions, dataset hash, layer, coordinate convention, random seed, package
versions, and Git commit when available. Tensor files are immutable outputs;
reruns use a new run directory.
