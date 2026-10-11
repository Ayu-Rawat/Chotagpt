# Chotagpt

> A small, locally trainable GPT-style language model implemented from scratch in PyTorch.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-not%20specified-lightgrey)](#license)

Chotagpt is an educational GPT implementation for experimenting with the complete language-model workflow: tokenization, causal self-attention, streamed pretraining, supervised fine-tuning, checkpoint management, text generation, and interactive chat. It runs locally with PyTorch and can use a CUDA GPU when one is available.

## Overview

This project contains a compact decoder-only Transformer rather than a wrapper around a hosted language-model API. The main model uses:

- GPT-2 BPE tokenization through `tiktoken`
- A vocabulary of 50,257 GPT-2 tokens
- A 256-token context window by default
- 6 Transformer blocks
- 6 causal self-attention heads per block
- A model width of 384
- Autoregressive next-token prediction

The repository supports two main training stages:

1. **Pretraining** on the streamed `Skylion007/openwebtext` dataset.
2. **Instruction fine-tuning** on the Alpaca instruction dataset, with loss applied to answer tokens rather than the user prompt.

The repository also includes reusable attention, normalization, positional-encoding, KV-cache, data-loader, and tokenizer exercises. The production training and chat path is centered on `model/gpt.py`, `train_openwebtext.py`, `sft_script.py`, and `chat.py`.

## Features

- **From-scratch GPT architecture** with token embeddings, learned positional embeddings, causal multi-head attention, feed-forward layers, residual connections, and layer normalization.
- **Streamed OpenWebText pretraining** through Hugging Face Datasets without downloading the entire corpus first.
- **Resume-friendly training** with `latest.pt`, step checkpoints, optimizer state, RNG state, document counters, and token counters.
- **Instruction fine-tuning** using Alpaca JSONL data and prompt masking so the model learns from assistant responses.
- **Local text generation** with temperature, top-k sampling, greedy decoding, and context cropping.
- **Interactive chat mode** using `<|user|>`, `<|assistant|>`, and `<|end|>` conversation markers.
- **Checkpoint diagnostics** for listing checkpoints, inspecting configuration and tensor counts, sampling text, and reviewing loss logs.
- **CPU and CUDA support** with optional automatic mixed precision, gradient accumulation, gradient checkpointing, and `torch.compile`.

## Demo

Run the interactive chat program after preparing an instruction-tuned checkpoint:

```text
loaded checkpoints/gpt_instruct.pt. type 'exit' to quit.

User: Explain what a neural network is.
Assistant: ...
```

To add a screenshot or recording to the repository, place it in `screenshots/` and update this section, for example:

```markdown
![Chotagpt interactive chat](screenshots/chat-demo.png)
```

## Requirements

- Python 3.10 or newer is recommended.
- PyTorch 2.0 or newer
- A CUDA-capable GPU is recommended for practical training, but the scripts fall back to CPU.
- Internet access is required the first time you download datasets or tokenizer assets.
- Several training configurations require substantial disk space and GPU memory.

Install the Python dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The listed dependencies are:

- `torch`
- `numpy`
- `torchtyping`
- `datasets`
- `tiktoken`

## Quick Start

### 1. Clone and enter the repository

```bash
git clone [REPOSITORY_URL]
cd Chotagpt
```

### 2. Create an environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 3. Run a small pretraining smoke test

The default training configuration is intended for a longer run. Override the step count and checkpoint interval when testing the pipeline:

```bash
python train_openwebtext.py --max_steps 30 --ckpt_every_steps 10
```

The script automatically selects CUDA when available. It writes checkpoints to `checkpoints/` and keeps a `latest.pt` file for convenient resuming.

### 4. Generate text from a checkpoint

```bash
python check_checkpoint.py --sample --prompt "The future of artificial intelligence" --n 80
```

Use greedy decoding instead of sampling:

```bash
python check_checkpoint.py --sample --greedy --prompt "The quick brown fox" --n 60
```

### 5. Inspect checkpoints

```bash
python check_checkpoint.py --list
python check_checkpoint.py --inspect
python check_checkpoint.py --all
```

The default checkpoint lookup is `checkpoints/latest.pt`. A specific checkpoint can be supplied as a positional argument:

```bash
python check_checkpoint.py checkpoints/ckpt_final_00018509.pt --inspect
```

## How to Use

### Pretraining

`train_openwebtext.py` streams text from Hugging Face, tokenizes it with GPT-2 BPE, creates fixed-length next-token prediction examples, and trains the model with AdamW.

Start a run:

```bash
python train_openwebtext.py
```

Useful options:

```powershell
python train_openwebtext.py `
  --max_steps 10000 `
  --micro_batch_size 4 `
  --grad_accum_steps 32 `
  --context_length 256 `
  --lr 0.0003 `
  --ckpt_every_steps 500 `
  --out_dir checkpoints
```

On Windows PowerShell, the backtick at the end of each line continues the command. The same options can be written on one line if preferred.

Resume from the latest checkpoint:

```bash
python train_openwebtext.py --resume
```

Resume from a particular checkpoint or directory:

```bash
python train_openwebtext.py --resume checkpoints/ckpt_00018500.pt
python train_openwebtext.py --resume checkpoints
```

Optional performance flags:

```bash
python train_openwebtext.py --compile
python train_openwebtext.py --no_grad_ckpt
```

`--compile` can reduce runtime after its initial compilation cost, while `--no_grad_ckpt` disables gradient checkpointing when debugging or when memory is not a concern.

### Instruction fine-tuning

The fine-tuning script expects:

- A base checkpoint at `checkpoints/ckpt_00018500.pt`
- Alpaca-format data at `dataset/alpaca.jsonl`

Download or regenerate the Alpaca JSONL file with:

```bash
python dataset/downlaod_alpaca.py
```

Then fine-tune:

```bash
python sft_script.py
```

The script:

- Loads the base model configuration and weights.
- Combines each `instruction` and optional `input` field into a user message.
- Formats conversations with `<|user|>`, `<|assistant|>`, and `<|end|>` markers.
- Masks prompt tokens with `-100`, so cross-entropy trains on answer tokens.
- Splits data into training and validation sets.
- Uses gradient accumulation, AdamW, warmup, cosine learning-rate decay, gradient clipping, and CUDA mixed precision when available.
- Saves `gpt_instruct_epoch1.pt`, `gpt_instruct_epoch2.pt`, `gpt_instruct_epoch3.pt`, and the final `gpt_instruct.pt`.

### Interactive chat

After `checkpoints/gpt_instruct.pt` exists:

```bash
python chat.py
```

Enter messages at the `User:` prompt. Type `exit` or `quit` to stop. Generation uses temperature `0.8`, top-k `40`, and up to 200 new tokens by default.

> Note: `chat.py` is configured for the instruction-tuned checkpoint format produced by `sft_script.py`. The OpenWebText checkpoint can be sampled with `check_checkpoint.py`.

### Loss history

If training output has been saved into `.log` files under `logs/`, inspect the recent loss values with:

```bash
python check_checkpoint.py --hist
```

To use another log directory or show a different number of lines:

```bash
python check_checkpoint.py --hist --log_dir logs --hist_tail 100
```

## How It Works

### Model

`model/gpt.py` defines the main `GPT` module. Tokens are mapped to learned embeddings and added to learned position embeddings. The resulting sequence passes through repeated pre-normalized Transformer blocks:

1. Layer normalization
2. Causal multi-head self-attention
3. Residual connection
4. Layer normalization
5. ReLU feed-forward network with dropout
6. Residual connection
7. Final layer normalization and vocabulary projection

The causal lower-triangular mask prevents each token from attending to future tokens.

### Tokenization

`data/tokenizer.py` provides a small wrapper around the GPT-2 encoding from `tiktoken`. It also retains a compatibility `CharacterTokenizer` for older checkpoints that stored `stoi` and `itos` mappings. `checkpoint.py` stores and restores tokenizer metadata alongside model configuration.

### Pretraining data flow

`train_openwebtext.py` uses an `IterableDataset` that:

1. Streams documents from `Skylion007/openwebtext`.
2. Tokenizes each document lazily.
3. Appends the GPT-2 end-of-text token.
4. Joins documents into a rolling token buffer.
5. Yields input and target sequences shifted by one token.

The data loader uses zero workers intentionally because streamed Hugging Face datasets and Windows process spawning are more reliable in this configuration.

### Checkpoints

Training checkpoints contain model weights, optimizer state, configuration, step and data counters, and CPU/CUDA RNG states. Checkpoints are written atomically through a temporary file before being moved into place. Older step checkpoints are rotated according to `keep_last_n`, while `latest.pt` provides a stable resume path.

## Project Structure

```text
Chotagpt/
├── README.md
├── requirements.txt
├── .gitignore
├── chat.py                       # Interactive instruction-tuned chat
├── generate.py                   # Reusable autoregressive generation helper
├── checkpoint.py                 # Generic checkpoint save/load utilities
├── check_checkpoint.py           # Checkpoint inspection and sampling CLI
├── train_openwebtext.py          # Streamed OpenWebText pretraining
├── sft_script.py                 # Alpaca supervised fine-tuning
├── diagnostic.py                 # Manual base-checkpoint generation diagnostic
├── data/
│   ├── tokenizer.py              # GPT-2 and legacy character tokenizers
│   ├── loader.py                 # Tensor batch creation helper
│   ├── dataset.py                # Text batch helper
│   ├── nlp_preprocessing.py      # Basic word-level preprocessing helper
│   ├── tokenizer_utils.py        # Greedy tokenization exercises
│   └── vocab.py                  # Vocabulary exercises
├── dataset/
│   ├── alpaca.jsonl              # Alpaca instruction data
│   └── downlaod_alpaca.py        # Dataset download script
├── model/
│   ├── gpt.py                    # Main GPT model
│   ├── attention.py              # Single-head causal attention
│   ├── multi_head_attention.py   # Multi-head attention component
│   ├── grouped_query_attention.py
│   ├── kv_cache.py               # KV cache and cached attention
│   ├── transformer.py             # Standalone Transformer block
│   ├── embeddings.py
│   ├── positional_encoding.py
│   ├── normalization.py
│   ├── rms_normalization.py
│   └── batch_normalization.py
└── checkpoints/                  # Generated .pt model files
```

`__pycache__/`, `.venv/`, and generated checkpoint files are local artifacts and are excluded by `.gitignore`. Checkpoint files can be hundreds of megabytes each; keep them out of normal source-control commits unless a release specifically requires model weights.

## Configuration

The default pretraining configuration is defined in `TrainConfig` in `train_openwebtext.py`:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `context_length` | 256 | Number of tokens processed per sequence |
| `model_dim` | 384 | Hidden size of the Transformer |
| `num_blocks` | 6 | Number of Transformer blocks |
| `num_heads` | 6 | Attention heads per block |
| `micro_batch_size` | 8 | Sequences per forward pass |
| `grad_accum_steps` | 32 | Micro-batches accumulated per optimizer step |
| `max_steps` | 200,000 | Maximum optimizer steps |
| `lr` | `3e-4` | Peak learning rate |
| `warmup_steps` | 2,000 | Linear warmup duration |
| `ckpt_every_steps` | 500 | Step-based checkpoint interval |
| `keep_last_n` | 3 | Number of rotated step checkpoints to retain |

The effective batch size is `micro_batch_size × grad_accum_steps`, or 256 sequences with the defaults. Reduce the batch size, context length, or accumulation settings when memory is limited.

## Limitations and Known Considerations

- This is an educational and experimental model, not a production-grade language model.
- The default model is small compared with modern GPT systems and should not be expected to match their quality, factuality, or reasoning ability.
- Generated text can be incorrect, repetitive, biased, or nonsensical. Always review outputs.
- The model has no built-in browsing, retrieval, tool use, or real-time knowledge.
- Pretraining depends on the availability and current format of the remote Hugging Face dataset.
- Training can be computationally expensive and may require a CUDA GPU with significant VRAM.
- The OpenWebText stream is shuffled with a buffer and checkpoint resume skips documents approximately; resumed training is not guaranteed to reproduce the exact same data order.
- `chat.py`, `sft_script.py`, and `check_checkpoint.py` use different checkpoint key conventions inherited from the project’s training stages. Use each script with the checkpoint type described in this README.
- The repository contains standalone component exercises in addition to the main GPT path; not every module under `data/` or `model/` is used by the end-to-end training scripts.
- The Alpaca dataset and OpenWebText are external data sources. Review their licenses and terms before redistribution or commercial use.

## Troubleshooting

### No checkpoint found

Create a checkpoint by running a short training job, or pass the path to an existing `.pt` file:

```bash
python train_openwebtext.py --max_steps 30
python check_checkpoint.py checkpoints/ckpt_00000030.pt --sample
```

### CUDA out-of-memory

Try reducing the per-forward-pass workload:

```bash
python train_openwebtext.py --micro_batch_size 2 --grad_accum_steps 32
```

You can also lower `--context_length`, disable compilation, or run on CPU for small tests.

### Dataset download or authentication errors

Confirm that the machine has internet access and that the `datasets` package is installed. The training script uses the `train` split of `Skylion007/openwebtext`; the fine-tuning download script uses `tatsu-lab/alpaca`.

### PowerShell execution policy

If activating `.venv` is blocked by local policy, install packages using the environment’s Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Roadmap

Potential future improvements include:

- Add automated unit tests for model shapes, masking, tokenization, generation, and checkpoint compatibility.
- Unify checkpoint schemas and expose a shared model-loading utility for all scripts.
- Add configurable model and fine-tuning settings through CLI arguments or a configuration file.
- Add evaluation metrics and a held-out benchmark for text quality.
- Add efficient KV-cache generation to the main GPT model for faster long-form sampling.
- Improve packaging, reproducibility, and experiment tracking.

## Contributing

Contributions are welcome. For changes to the model or training behavior:

1. Create a feature branch.
2. Make a focused change.
3. Update the README or relevant inline documentation.
4. Run a small smoke test before opening a pull request:

   ```bash
   python train_openwebtext.py --max_steps 1 --ckpt_every_steps 1
   python check_checkpoint.py --inspect
   ```

5. Include the command used to validate the change and describe any hardware requirements.

Please do not commit API keys, `.env` files, virtual environments, `__pycache__/`, or large generated checkpoints unless the change specifically requires them.

## License

No license file is currently present in the repository. Until a license is added, all rights are reserved by the copyright holder by default.

If you intend others to use, modify, or redistribute this project, add a `LICENSE` file and replace the badge and this section with the selected license terms.
