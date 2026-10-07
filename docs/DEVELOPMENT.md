# Development guide

## Repository conventions

The project uses small, focused Python modules. Course exercises generally
follow this shape:

```python
class Solution:
    def method_name(...):
        ...
```

PyTorch modules subclass `torch.nn.Module`; NumPy exercises use NumPy arrays
and often return values rounded to a fixed number of decimal places. Preserve
the existing public method signatures when adding or correcting an exercise.

## Adding a new exercise

1. Put the implementation in the package that matches its topic:
   `data/` or `model/`.
2. Add a focused docstring or short comment only where behavior is not
   obvious.
3. If the module is intended as a package-level exercise export, add it to
   that package's `__init__.py`; otherwise use a direct module import.
4. Add a usage example to the relevant documentation page when the public
   API is user-facing.
5. Run the import smoke test and a small deterministic example.

## Validation

There is no test runner configuration or checked-in test directory. The
minimum validation is:

```powershell
python -m compileall data model train.py generate.py checkpoint.py run_gpt.py
python -c "import data, model, checkpoint; print('imports ok')"
```

For the GPT path, also run a shape check:

```powershell
python -c "import torch; from model.gpt import GPT; m=GPT(32,8,16,1,4); print(m(torch.zeros((2,8),dtype=torch.long)).shape)"
```

Expected output is `torch.Size([2, 8, 32])`.

Use the environment's interpreter explicitly on Windows when needed:

```powershell
.\.venv\Scripts\python.exe -m compileall data model train.py generate.py checkpoint.py run_gpt.py
```

## Working with randomness

Many modules call `torch.manual_seed(0)` internally. This makes course
answers repeatable but can reset the caller's global random state. New
production-oriented code should avoid resetting global state inside `forward`
or helper methods unless deterministic behavior is an explicit requirement.

## Common integration mistakes

- Pass integer token IDs (`torch.long`) to `nn.Embedding`.
- Keep token IDs within `[0, vocab_size)`.
- Ensure `len(data) > context_length` before sampling training windows.
- Use a context whose second dimension does not exceed the GPT context length.
- Keep the model's vocabulary size, tokenizer mapping, and decoder mapping
  consistent.
- Put the model in evaluation mode before generation if dropout is present:
  `model.eval()`.
- Package initializers expose modules, so import exercise classes from their
  concrete modules, such as `from data.vocab import Solution`.

## Suggested future work

- Add a device-selection option and checkpoint resume support.
- Add tests for tensor shapes, causal masking, deterministic behavior, and
  edge cases such as short datasets.
- Document and pin a supported Python version range.
