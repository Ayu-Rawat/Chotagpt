# NeetCode GPT

This repository contains an educational GPT implementation with a unified
training, checkpointing, and text-generation pipeline.

## Project structure

```text
model/          Attention, transformer, and GPT architecture
data/           Vocabulary, tokenization, preprocessing, and batching
train.py        Reusable AdamW training helper
generate.py     Reusable autoregressive generation helper
checkpoint.py   Checkpoint save/load helpers
run_gpt.py      End-to-end training and generation entry point
```

## Quick start

```bash
pip install -r requirements.txt
python run_gpt.py
```

Put training text in `input.txt` to replace the built-in fallback sample.
Use `python run_gpt.py --help` to view configuration options. The script
trains the model, writes `gpt_checkpoint.pt`, reloads it, and generates 100
tokens.

See the [documentation](docs/README.md) for setup, architecture, examples,
and development notes.
