# Concept-vector J-space occupancy v2

## Scientific question

For a concept probe direction \(w_{c,l}\) at layer \(l\), v2 asks for the
smallest sparse non-negative combination of token J-directions that remains
more informative than a matched random dictionary:

\[
w_{J,c,l}(K)=\sum_{t\in S_K}\alpha_t A_l[t,:],
\qquad
\alpha_t\ge 0,
\qquad
A_l=U_{\mathrm{eff}}J_l.
\]

This changes the target in the Workspace-style J-space occupancy experiment
from a residual activation to a concept vector. It does not change the
dictionary, coordinate, sparse non-negativity constraint, or random-control
logic. All targets and reconstructions are in block-output `resid_post`
coordinates.

## End-to-end protocol

1. Reuse the identity-checked Qwen3.5-4B activation, primary probe, and fitted
   J-lens artifacts. The model, tokenizer, BOS policy, data revision, lens
   content hash, layer, and coordinate must agree.
2. Hold the primary probe's train-selected logistic `C` fixed. Create four
   stratified group-bootstrap directions from train+validation groups only.
   The test split is not accessed.
3. For every target direction, scan the full token dictionary at every pursuit
   step. Select the token with the largest positive residual cosine and refit
   the active non-negative coefficients with the versioned projected-gradient
   solver `nonnegative_gradient_pursuit_v2`.
4. Repeat the same pursuit against five Gaussian dictionaries whose individual
   atom norms match the real J-directions. The full-cardinality control is
   primary; a 1/16-cardinality control is retained as a sensitivity analysis.
5. At each integer \(k\), record normalized squared reconstruction error and
   marginal gain. `crossing_k` is the first candidate step whose real marginal
   gain does not exceed the median matched-random gain.
   `k_selected_before_crossing = crossing_k - 1` is the number of preceding
   atoms that remain supported by the control comparison. A curve with no
   crossing is right-censored at `k_max`.
6. Save the reconstructed `w_J` and residual `w_nonJ = w - w_J` at every
   reporting point and at `k_selected_before_crossing`. `w_nonJ` is a
   reconstruction residual, not an orthogonal complement.
7. Require consistency across probe bootstraps and inspect both the raw and
   RMSNorm-weighted unembedding conventions before choosing a layer and \(K\)
   for the later causal intervention experiment.

The paper names Gradient Pursuit but does not publish the sparse solver
implementation or every normalization detail. Therefore v2 is a versioned,
documented-method reproduction, not a bitwise claim about the authors' private
code.

## Registered scope

- Layers: 8, 12, 16, 20, 24, 28
- Concepts: admiration, approval, curiosity, disapproval, gratitude, love,
  optimism
- Target signs: positive and negative
- Conventions: raw and RMSNorm-weighted
- Selection rule: positive residual cosine
- Primary maximum support: 64
- Bootstrap maximum support: 25
- Probe directions: one primary plus four group-bootstrap replicates
- Random controls: five seeds at full and 1/16 dictionary cardinality
- Expected decomposition combinations: 840

The configuration is
`Concept_intervention/configs/qwen35_4b_concept_occupancy_v2.yaml`. The
dependency chain is submitted with
`Concept_intervention/scripts/submit_qwen35_4b_concept_occupancy_v2.sh`.
Outputs are written under
`artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2/`.

## Interpretation gate before intervention

The later intervention should use a layer and \(K\) only after the v2 results
show:

- a stable `k_selected_before_crossing` distribution across bootstrap probes;
- a meaningful reconstruction gain over the full-cardinality random control;
- no conclusion that depends solely on one coordinate convention; and
- semantically coherent selected tokens for the positive concept direction.

That causal experiment must separately include signed strengths and matched
full-vector, J-component, non-J residual, and random-direction controls.
