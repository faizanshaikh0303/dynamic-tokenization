# Dynamic Tokenization with Model Feedback

This project explores a closed-loop tokenizer that updates BPE merge rules based on model feedback (gradient magnitudes, token-level loss, and attention entropy). The goal is to determine whether tokenization can be optimized **during training** to improve compression and minimize loss.

## Features
- Baseline BPE tokenizer (HuggingFace `tokenizers`)
- Gradient + attention + loss–based feedback signals
- Dynamic merge updater
- Tiny transformer model for experimentation
- Closed-loop training harness
- Metrics + tokenizer stability checks

## Quickstart
pip install -r requirements.txt
python experiments/run_dynamic_bpe.py

## Status
Early prototype — dynamic update loop partially implemented.
