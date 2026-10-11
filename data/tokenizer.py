"""Tokenizer helpers built on top of GPT-2 BPE from tiktoken."""

from __future__ import annotations

from typing import Any, Mapping

try:
    import tiktoken
except ImportError as exc:  # pragma: no cover - dependency error path
    raise ImportError(
        "tiktoken is required for GPT-2 sub-word tokenization. Install the "
        "project requirements before running the GPT scripts."
    ) from exc


class GPT2Tokenizer:
    """Thin wrapper around the GPT-2 encoding used by GPT-2/3 style models."""

    def __init__(self, encoding_name: str = "gpt2") -> None:
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.encoding.decode(token_ids)

    def get_state(self) -> dict[str, str]:
        return {"kind": "tiktoken", "encoding_name": self.encoding_name}


class CharacterTokenizer:
    """Compatibility tokenizer for older checkpoints that stored char maps."""

    def __init__(self, stoi: Mapping[str, int], itos: Mapping[int, str]) -> None:
        self.stoi = dict(stoi)
        self.itos = {int(index): token for index, token in itos.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)

    def get_state(self) -> dict[str, Any]:
        return {"kind": "character", "stoi": dict(self.stoi), "itos": dict(self.itos)}


def build_tokenizer(state: Mapping[str, Any] | None = None) -> GPT2Tokenizer | CharacterTokenizer:
    """Construct a tokenizer from checkpoint metadata or use GPT-2 by default."""

    if state is None:
        return GPT2Tokenizer()

    kind = state.get("kind") or state.get("type")
    if kind in {"character", "legacy-character"}:
        return CharacterTokenizer(state.get("stoi", {}), state.get("itos", {}))

    encoding_name = state.get("encoding_name", "gpt2")
    return GPT2Tokenizer(encoding_name)


class Solution(GPT2Tokenizer):
    """Legacy alias kept so existing imports continue to work."""
from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []
        for _ in range(num_merges):
            if len(tokens) < 2:
                break
            # Count adjacent pair frequencies
            pairs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break

            # Find most frequent pair (tiebreak: lexicographically smallest)
            best_count = max(pairs.values())
            candidates = sorted(p for p, c in pairs.items() if c == best_count)
            best = candidates[0]

            merges.append([best[0], best[1]])

            # Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                    new_tokens.append(best[0] + best[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return merges