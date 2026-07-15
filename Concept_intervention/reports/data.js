// Lightweight, reviewable summary derived from the immutable v1 artifact.
// Add future Concept Intervention runs with another register("concept", {...}) call.
window.JLensReportRegistry.register("concept", {
  "id": "qwen35_4b_goemotions_v1",
  "title": "GoEmotions 7-concept baseline",
  "shortTitle": "GoEmotions v1",
  "status": "exploratory",
  "reportedAt": "2026-07-14",
  "question": "Can seven emotion concepts be linearly decoded from Qwen3.5-4B resid_post activations, and do their probe directions retrieve interpretable token J-directions?",
  "summary": "All 42 held-out probes reproduce exactly from the saved activations and raw probe vectors. Linear decodability is established, but v1 has no random, non-J, or causal intervention controls.",
  "layers": [
    8,
    12,
    16,
    20,
    24,
    28
  ],
  "probeResults": [
    {
      "concept": "admiration",
      "meanAuc": 0.881485,
      "minAuc": 0.877103,
      "maxAuc": 0.887356,
      "meanAp": 0.520305,
      "bestLayer": 28,
      "gate": "promising",
      "gateNote": "Semantic tokens are clear, but cross-layer stability still needs random controls.",
      "byLayer": [
        {
          "layer": 8,
          "auc": 0.879001,
          "ap": 0.523725,
          "balancedAccuracy": 0.808874,
          "chosenC": 0.0003
        },
        {
          "layer": 12,
          "auc": 0.883081,
          "ap": 0.521708,
          "balancedAccuracy": 0.806725,
          "chosenC": 0.0003
        },
        {
          "layer": 16,
          "auc": 0.883158,
          "ap": 0.521244,
          "balancedAccuracy": 0.797414,
          "chosenC": 0.0003
        },
        {
          "layer": 20,
          "auc": 0.877103,
          "ap": 0.52166,
          "balancedAccuracy": 0.798054,
          "chosenC": 0.0003
        },
        {
          "layer": 24,
          "auc": 0.879208,
          "ap": 0.512674,
          "balancedAccuracy": 0.79775,
          "chosenC": 0.0003
        },
        {
          "layer": 28,
          "auc": 0.887356,
          "ap": 0.52082,
          "balancedAccuracy": 0.811414,
          "chosenC": 0.0003
        }
      ]
    },
    {
      "concept": "approval",
      "meanAuc": 0.785117,
      "minAuc": 0.778036,
      "maxAuc": 0.795089,
      "meanAp": 0.247631,
      "bestLayer": 20,
      "gate": "weak",
      "gateNote": "AUC, AP, and token stability are weakest; do not advance to intervention yet.",
      "byLayer": [
        {
          "layer": 8,
          "auc": 0.779668,
          "ap": 0.259011,
          "balancedAccuracy": 0.706244,
          "chosenC": 0.0001
        },
        {
          "layer": 12,
          "auc": 0.792417,
          "ap": 0.26315,
          "balancedAccuracy": 0.71032,
          "chosenC": 0.0001
        },
        {
          "layer": 16,
          "auc": 0.786925,
          "ap": 0.257889,
          "balancedAccuracy": 0.700531,
          "chosenC": 0.0001
        },
        {
          "layer": 20,
          "auc": 0.795089,
          "ap": 0.2501,
          "balancedAccuracy": 0.712174,
          "chosenC": 0.0001
        },
        {
          "layer": 24,
          "auc": 0.778569,
          "ap": 0.229295,
          "balancedAccuracy": 0.700099,
          "chosenC": 0.0001
        },
        {
          "layer": 28,
          "auc": 0.778036,
          "ap": 0.226338,
          "balancedAccuracy": 0.697034,
          "chosenC": 0.0001
        }
      ]
    },
    {
      "concept": "curiosity",
      "meanAuc": 0.926016,
      "minAuc": 0.912734,
      "maxAuc": 0.93285,
      "meanAp": 0.438443,
      "bestLayer": 20,
      "gate": "confounded",
      "gateNote": "May encode question-answer form; shortcut-controlled data are required.",
      "byLayer": [
        {
          "layer": 8,
          "auc": 0.912734,
          "ap": 0.408604,
          "balancedAccuracy": 0.842489,
          "chosenC": 0.0003
        },
        {
          "layer": 12,
          "auc": 0.924817,
          "ap": 0.439863,
          "balancedAccuracy": 0.85996,
          "chosenC": 0.0003
        },
        {
          "layer": 16,
          "auc": 0.928484,
          "ap": 0.463004,
          "balancedAccuracy": 0.868302,
          "chosenC": 0.0003
        },
        {
          "layer": 20,
          "auc": 0.93285,
          "ap": 0.454851,
          "balancedAccuracy": 0.865362,
          "chosenC": 0.0001
        },
        {
          "layer": 24,
          "auc": 0.927624,
          "ap": 0.432496,
          "balancedAccuracy": 0.85859,
          "chosenC": 0.0001
        },
        {
          "layer": 28,
          "auc": 0.929589,
          "ap": 0.431843,
          "balancedAccuracy": 0.863989,
          "chosenC": 0.0001
        }
      ]
    },
    {
      "concept": "disapproval",
      "meanAuc": 0.850579,
      "minAuc": 0.846023,
      "maxAuc": 0.8576,
      "meanAp": 0.245417,
      "bestLayer": 20,
      "gate": "confounded",
      "gateNote": "Token rankings are dominated by negation; match negation structure before intervention.",
      "byLayer": [
        {
          "layer": 8,
          "auc": 0.8461,
          "ap": 0.245787,
          "balancedAccuracy": 0.759544,
          "chosenC": 0.0001
        },
        {
          "layer": 12,
          "auc": 0.846023,
          "ap": 0.241839,
          "balancedAccuracy": 0.762786,
          "chosenC": 0.0001
        },
        {
          "layer": 16,
          "auc": 0.854289,
          "ap": 0.245749,
          "balancedAccuracy": 0.780813,
          "chosenC": 0.0001
        },
        {
          "layer": 20,
          "auc": 0.8576,
          "ap": 0.253735,
          "balancedAccuracy": 0.763427,
          "chosenC": 0.0001
        },
        {
          "layer": 24,
          "auc": 0.851679,
          "ap": 0.248733,
          "balancedAccuracy": 0.767831,
          "chosenC": 0.0001
        },
        {
          "layer": 28,
          "auc": 0.847785,
          "ap": 0.236656,
          "balancedAccuracy": 0.75707,
          "chosenC": 0.0001
        }
      ]
    },
    {
      "concept": "gratitude",
      "meanAuc": 0.966747,
      "minAuc": 0.964287,
      "maxAuc": 0.971708,
      "meanAp": 0.845185,
      "bestLayer": 28,
      "gate": "strong",
      "gateNote": "Held-out metrics and gratitude-related tokens are both stable.",
      "byLayer": [
        {
          "layer": 8,
          "auc": 0.964667,
          "ap": 0.836445,
          "balancedAccuracy": 0.906793,
          "chosenC": 0.0003
        },
        {
          "layer": 12,
          "auc": 0.964287,
          "ap": 0.826492,
          "balancedAccuracy": 0.903771,
          "chosenC": 0.0003
        },
        {
          "layer": 16,
          "auc": 0.965678,
          "ap": 0.828073,
          "balancedAccuracy": 0.91061,
          "chosenC": 0.0003
        },
        {
          "layer": 20,
          "auc": 0.967033,
          "ap": 0.855127,
          "balancedAccuracy": 0.919863,
          "chosenC": 0.001
        },
        {
          "layer": 24,
          "auc": 0.967107,
          "ap": 0.853547,
          "balancedAccuracy": 0.914459,
          "chosenC": 0.001
        },
        {
          "layer": 28,
          "auc": 0.971708,
          "ap": 0.871429,
          "balancedAccuracy": 0.918708,
          "chosenC": 0.001
        }
      ]
    },
    {
      "concept": "love",
      "meanAuc": 0.921735,
      "minAuc": 0.907713,
      "maxAuc": 0.93548,
      "meanAp": 0.492954,
      "bestLayer": 28,
      "gate": "strong",
      "gateNote": "Held-out metrics are strong and token alignment is the most direct.",
      "byLayer": [
        {
          "layer": 8,
          "auc": 0.911974,
          "ap": 0.470039,
          "balancedAccuracy": 0.83751,
          "chosenC": 0.0003
        },
        {
          "layer": 12,
          "auc": 0.907713,
          "ap": 0.441555,
          "balancedAccuracy": 0.822705,
          "chosenC": 0.0003
        },
        {
          "layer": 16,
          "auc": 0.918287,
          "ap": 0.476916,
          "balancedAccuracy": 0.83905,
          "chosenC": 0.0003
        },
        {
          "layer": 20,
          "auc": 0.929643,
          "ap": 0.516944,
          "balancedAccuracy": 0.853369,
          "chosenC": 0.0003
        },
        {
          "layer": 24,
          "auc": 0.927314,
          "ap": 0.511446,
          "balancedAccuracy": 0.858156,
          "chosenC": 0.0003
        },
        {
          "layer": 28,
          "auc": 0.93548,
          "ap": 0.540827,
          "balancedAccuracy": 0.867729,
          "chosenC": 0.0003
        }
      ]
    },
    {
      "concept": "optimism",
      "meanAuc": 0.883257,
      "minAuc": 0.877198,
      "maxAuc": 0.889518,
      "meanAp": 0.330365,
      "bestLayer": 20,
      "gate": "strong",
      "gateNote": "Hope-related tokens recur across layers.",
      "byLayer": [
        {
          "layer": 8,
          "auc": 0.877198,
          "ap": 0.340177,
          "balancedAccuracy": 0.787503,
          "chosenC": 0.0001
        },
        {
          "layer": 12,
          "auc": 0.881585,
          "ap": 0.338364,
          "balancedAccuracy": 0.783067,
          "chosenC": 0.0001
        },
        {
          "layer": 16,
          "auc": 0.877583,
          "ap": 0.349448,
          "balancedAccuracy": 0.784319,
          "chosenC": 0.0001
        },
        {
          "layer": 20,
          "auc": 0.889518,
          "ap": 0.328611,
          "balancedAccuracy": 0.798121,
          "chosenC": 0.0001
        },
        {
          "layer": 24,
          "auc": 0.884698,
          "ap": 0.314498,
          "balancedAccuracy": 0.797069,
          "chosenC": 0.0001
        },
        {
          "layer": 28,
          "auc": 0.888962,
          "ap": 0.311092,
          "balancedAccuracy": 0.808754,
          "chosenC": 0.0001
        }
      ]
    }
  ],
  "layerMeanAuc": [
    {
      "layer": 8,
      "auc": 0.88162
    },
    {
      "layer": 12,
      "auc": 0.885703
    },
    {
      "layer": 16,
      "auc": 0.887772
    },
    {
      "layer": 20,
      "auc": 0.892691
    },
    {
      "layer": 24,
      "auc": 0.888028
    },
    {
      "layer": 28,
      "auc": 0.891274
    }
  ],
  "alignments": [
    {
      "concept": "admiration",
      "layer": 28,
      "tokens": [
        {
          "token": "Excellent",
          "score": 0.1182
        },
        {
          "token": "Excellent",
          "score": 0.1153
        },
        {
          "token": "superb",
          "score": 0.1143
        }
      ],
      "gate": "promising",
      "note": "Semantic tokens are clear, but cross-layer stability still needs random controls."
    },
    {
      "concept": "approval",
      "layer": 28,
      "tokens": [
        {
          "token": "Ag",
          "score": 0.1127
        },
        {
          "token": "Ag",
          "score": 0.1043
        },
        {
          "token": "Tuttavia",
          "score": 0.1011
        }
      ],
      "gate": "weak",
      "note": "AUC, AP, and token stability are weakest; do not advance to intervention yet."
    },
    {
      "concept": "curiosity",
      "layer": 28,
      "tokens": [
        {
          "token": "answer [zh]",
          "score": 0.1791
        },
        {
          "token": "the answer [zh]",
          "score": 0.1634
        },
        {
          "token": "answers",
          "score": 0.161
        }
      ],
      "gate": "confounded",
      "note": "May encode question-answer form; shortcut-controlled data are required."
    },
    {
      "concept": "disapproval",
      "layer": 28,
      "tokens": [
        {
          "token": "nor",
          "score": 0.1991
        },
        {
          "token": "Nor",
          "score": 0.1726
        },
        {
          "token": "Nor",
          "score": 0.1553
        }
      ],
      "gate": "confounded",
      "note": "Token rankings are dominated by negation; match negation structure before intervention."
    },
    {
      "concept": "gratitude",
      "layer": 28,
      "tokens": [
        {
          "token": "appreciate",
          "score": 0.1076
        },
        {
          "token": "appreciated",
          "score": 0.1016
        },
        {
          "token": "thanks [zh]",
          "score": 0.0984
        }
      ],
      "gate": "strong",
      "note": "Held-out metrics and gratitude-related tokens are both stable."
    },
    {
      "concept": "love",
      "layer": 28,
      "tokens": [
        {
          "token": "love",
          "score": 0.2834
        },
        {
          "token": "Love",
          "score": 0.2609
        },
        {
          "token": "Love",
          "score": 0.254
        }
      ],
      "gate": "strong",
      "note": "Held-out metrics are strong and token alignment is the most direct."
    },
    {
      "concept": "optimism",
      "layer": 28,
      "tokens": [
        {
          "token": "Hope",
          "score": 0.1689
        },
        {
          "token": "hope",
          "score": 0.1662
        },
        {
          "token": "Hope",
          "score": 0.1655
        }
      ],
      "gate": "strong",
      "note": "Hope-related tokens recur across layers."
    }
  ],
  "counts": {
    "examples": 53861,
    "taskRows": 377027,
    "train": 43112,
    "validation": 5370,
    "test": 5379,
    "concepts": 7,
    "probes": 42
  },
  "provenance": [
    [
      "Model",
      "Qwen/Qwen3.5-4B"
    ],
    [
      "Model revision",
      "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    ],
    [
      "Coordinate",
      "resid_post / block output, D = 2560"
    ],
    [
      "Dataset",
      "GoEmotions @ add492243ff905527e67aeb8b80c082af02207c3"
    ],
    [
      "Seed",
      "42"
    ],
    [
      "Lens fit",
      "1,000 prompts; float32 storage; target L31"
    ],
    [
      "Fit prompt SHA-256",
      "4dcb7c85ecbcbaade5fd029019b8f03d9ed373deeb648981aa75afc98e5eaa6e"
    ],
    [
      "Manifest SHA-256",
      "c86b1b3ebeed162f8788feebe3f5026f0b4b9539e27f704e6475728098a86f7a"
    ],
    [
      "Recorded Git commit",
      "null"
    ]
  ],
  "limitations": [
    "alignment files do not contain matched random controls or J/non-J decomposition",
    "intervention stage was not run; no causal steering claim is supported",
    "approval is not a reliable semantic alignment result",
    "curiosity and disapproval may reflect question/answer and negation shortcuts",
    "git_commit and lens_revision are null in the immutable v1 manifest"
  ],
  "sourceReport": "../Concept_intervention/reports/qwen35_4b_goemotions_v1.md"
});
