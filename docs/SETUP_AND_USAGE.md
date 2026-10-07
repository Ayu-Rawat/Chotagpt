# Setup and usage

## Requirements

- Python (the project does not currently pin a Python minor version)
- PyTorch `>=2.0.0`
- NumPy `>=1.24.0`
- torchtyping `>=0.1.4`

Dependencies are declared in [`requirements.txt`](../requirements.txt).

## Windows setup

From the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either enable scripts for the current user
or invoke the environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The repository's existing `.gptvenv` can also be used if it is already
configured:

```powershell
.\.gptvenv\Scripts\python.exe -m pip install -r requirements.txt
```

## Import smoke test

Run this from the repository root after installation:

```powershell
python -c "import data, model, checkpoint; print('imports ok')"
```

## Train a GPT model

`train.py` contains `train.Solution.train`. It accepts an instantiated
PyTorch model and a one-dimensional integer token tensor. The helper samples
random context windows, trains with AdamW and cross-entropy, and returns the
final loss without resetting the global RNG or rounding it.

```python
import torch
from model.gpt import GPT
from train import Solution as Trainer

vocab_size = 32
context_length = 8
model = GPT(
    vocab_size=vocab_size,
    context_length=context_length,
    model_dim=16,
    num_blocks=1,
    num_heads=4,
)

token_ids = torch.randint(0, vocab_size, (256,), dtype=torch.long)
final_loss = Trainer().train(
    model=model,
    data=token_ids,
    epochs=10,
    context_length=context_length,
    batch_size=8,
    lr=1e-3,
)
print(final_loss)
```

The input data must contain more tokens than `context_length`; otherwise the
random window range is invalid.

## Unified pipeline

`run_gpt.py` is the recommended entry point:

```powershell
python run_gpt.py --input input.txt --epochs 100
```

It reads `input.txt` (or uses a built-in sample), builds character mappings,
encodes the data, trains a configurable GPT, saves `gpt_checkpoint.pt`,
reloads the checkpoint, and generates 100 tokens. Useful options include
`--epochs`, `--batch-size`, `--learning-rate`, `--context-length`,
`--model-dim`, `--num-blocks`, `--num-heads`, `--new-tokens`, and `--prompt`.
Use fewer blocks/dimensions and fewer epochs for a quick local smoke run.

## Generate text

`generate.py` contains `generate.Solution.generate`. It repeatedly feeds the
latest context through a model, samples the next token, and decodes it with an
integer-to-character mapping.

```python
import torch
from model.gpt import GPT
from generate import Solution as Generator

context_length = 8
model = GPT(vocab_size=32, context_length=context_length,
            model_dim=16, num_blocks=1, num_heads=4)
context = torch.zeros((1, 1), dtype=torch.long)
int_to_char = {i: chr(97 + i) for i in range(32)}

text = Generator().generate(
    model=model,
    new_chars=20,
    context=context,
    context_length=context_length,
    int_to_char=int_to_char,
)
print(text)
```

For future integrations, `checkpoint.save_checkpoint` stores weights,
vocabulary mappings, and model metadata. `checkpoint.load_checkpoint`
returns that mapping on the CPU so a Web UI or agent can reconstruct `GPT`
with `GPT(**checkpoint["config"])` and load
`checkpoint["model_state_dict"]`.

## Data pipeline examples

Character vocabulary:

```python
from data.vocab import Solution as Vocabulary

stoi, itos = Vocabulary().build_vocab("hello world")
ids = Vocabulary().encode("hello", stoi)
text = Vocabulary().decode(ids, itos)
```

Random next-token batches:

```python
import torch
from data.loader import Solution as BatchLoader

x, y = BatchLoader().create_batches(
    torch.arange(100, dtype=torch.long),
    context_length=8,
    batch_size=4,
)
```

`data.dataset.Solution.batch_loader` provides the same next-token idea for
whitespace-tokenized strings and returns lists of string windows. The
tokenizer, vocabulary, preprocessing, and tokenizer utility modules are
independent exercises; they are not automatically wired into `train.py`.

## Current limitations

- No dataset download or preprocessing command is included.
- No automated tests are checked in.
- Some standalone exercise modules retain course-oriented rounding or
  initialization behavior; the unified `run_gpt.py` path does not reset the
  global RNG or round model logits.
