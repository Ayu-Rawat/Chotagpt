# Architecture

## High-level flow

The intended GPT path is:

1. Convert raw text to token IDs with `data.vocab` or another tokenizer.
2. Sample `(x, y)` next-token windows with `data.loader`.
3. Embed token IDs and positions in `model.gpt.GPT`.
4. Process the sequence through transformer blocks:
   pre-layer normalization, causal multi-head self-attention, residual
   connection, feed-forward network, and a second residual connection.
5. Project each hidden state to vocabulary logits.
6. Train with `train.Solution.train` and
   `torch.nn.functional.cross_entropy`.
7. Save/reload model state with `checkpoint.py`.
8. Autoregressively sample and decode with `generate.Solution.generate`.

## Repository map

### `data/`

- `vocab.py`: deterministic character-to-integer and integer-to-character
  mappings.
- `tokenizer.py`: simple iterative byte-pair-style adjacent character merges.
- `tokenizer_utils.py`: greedy longest-match tokenization, token counts, and
  fertility scores.
- `nlp_preprocessing.py`: builds a sorted word vocabulary and pads encoded
  sentences.
- `loader.py`: samples tensor context/target pairs for next-token training.
- `dataset.py`: samples whitespace-tokenized string context/target pairs.
- `__init__.py`: module-level exports without wildcard namespace collisions.

### `model/`

- `gpt.py`: complete GPT-style model with token embeddings, positional
  embeddings, transformer blocks, final layer normalization, and vocabulary
  projection.
- `attention.py`: single-head causal self-attention.
- `multi_head_attention.py`: standalone multi-head attention.
- `transformer.py`: standalone transformer block with attention, feed-forward
  network, layer normalization, residual paths, and dropout.
- `normalization.py`, `batch_normalization.py`, `rms_normalization.py`:
  normalization exercises implemented with NumPy or PyTorch.
- `embeddings.py`: NumPy embedding lookup.
- `positional_encoding.py`: sinusoidal positional encoding.
- `kv_cache.py`: append-only key/value cache and cached causal attention.
- `grouped_query_attention.py`: grouped-query attention with fewer K/V heads.
- `__init__.py`: module-level exports without wildcard namespace collisions.

### Root pipeline modules

- `run_gpt.py`: reads text, builds a character vocabulary, trains GPT,
  checkpoints it, reloads it, and generates text.
- `checkpoint.py`: persists model state, vocabulary mappings, and model
  reconstruction metadata.
- `train.py`: reusable AdamW next-token training helper.
- `generate.py`: reusable autoregressive sampling helper.

## Tensor shapes in the GPT path

For batch size `B`, sequence length `T`, model dimension `D`, and vocabulary
size `V`:

| Stage | Shape |
| --- | --- |
| Input token IDs | `(B, T)` |
| Token/position embeddings | `(B, T, D)` |
| Attention Q/K/V per head | `(B, T, D / num_heads)` |
| Transformer output | `(B, T, D)` |
| Vocabulary logits | `(B, T, V)` |
| Flattened training logits | `(B*T, V)` |
| Flattened targets | `(B*T)` |

`model_dim` should be divisible by `num_heads`. The context length passed to
`GPT` is also the maximum sequence length supported by its positional
embedding table.

## Design and behavior notes

- Attention is causal: future positions are masked with `-inf`.
- The unified GPT path preserves floating-point logits for normal PyTorch
  optimization and sampling.
- `GPT` uses `nn.Sequential` for its transformer blocks.
- `KVCache` is separate from `GPT`; it is an exercise/helper and is not
  automatically used by `generate.py`.
- Package initializers export modules rather than wildcard symbols, avoiding
  collisions between the many exercise classes named `Solution`.
