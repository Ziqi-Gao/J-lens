# Pinned ITI source

This workspace adapts the core Inference-Time Intervention (ITI) routines from
[`likenneth/honest_llama`](https://github.com/likenneth/honest_llama), pinned at
commit `2c6b2179be7b5aa8f0a171688cf9e01b812ca327`.

The attributed adaptation is in
`src/jlens_workspace/_vendor/honest_llama_core.py`. It preserves the original
probe training, validation-accuracy head ranking, mass-mean directions,
projection-standard-deviation scaling, and last-token addition. The Qwen3.5
adapter changes only data layout, head dimension, batch handling, and module
resolution. The upstream MIT license is retained in this directory.
