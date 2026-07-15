# Experiment report frontend

Open [`index.html`](index.html) directly, or serve the repository root for a
stable local URL:

```bash
python3 -m http.server 8000
```

Then visit `http://127.0.0.1:8000/reports/`.

The report has no external JavaScript, CSS, font, or charting dependency. All
plots are accessible inline SVG generated from small, reviewable summaries.
Large immutable artifacts remain below ignored `artifacts/` paths.

## Structure

```text
reports/
├── index.html                  shared report shell
└── assets/
    ├── registry.js             run registry
    ├── report.css              shared visual system and print layout
    └── report.js               renderers and SVG chart primitives
Concept_intervention/reports/
├── data.js                     Concept run summaries
└── *.md                        narrative source reports
J_space/reports/
├── data.js                     J-space run summaries
└── *.md                        narrative source reports
```

Direction-specific results remain independent: Concept data never imports a
J-space artifact, and J-space data never imports a Concept artifact. The shared
frontend only reads their registered summaries.

## Add a new run

1. Keep the immutable scientific output in a new ignored artifact directory.
2. Add its narrative Markdown report under the corresponding direction.
3. Append one `JLensReportRegistry.register(...)` object to that direction's
   `data.js`. Give it a stable, unique `id`; do not edit an existing run.
4. Record model/tokenizer/lens/data revisions, coordinate, seed, dtypes,
   manifest hash, Git commit, controls, and limitations.
5. Use raw per-layer values in the summary. The frontend computes display
   layout but does not invent or smooth measurements.
6. Run the report contract test and JavaScript syntax checks.

The run selector is populated automatically from the registry. New report
types should be implemented as reusable renderer functions in `report.js`, not
as copied HTML blocks inside a direction data module.
