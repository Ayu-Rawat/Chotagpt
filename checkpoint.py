"""Checkpoint persistence for trained GPT models."""

from pathlib import Path
from typing import Any, Mapping

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    tokenizer_state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Save model weights and the metadata required to reconstruct the model."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "tokenizer_state": dict(tokenizer_state),
        "config": dict(config),
    }
    torch.save(checkpoint, Path(path))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint created by :func:`save_checkpoint` on the CPU."""
    checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} must contain a mapping")
    required_keys = {"model_state_dict", "config"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Checkpoint at {path} is missing: {missing}")

    if "tokenizer_state" not in checkpoint:
        if "stoi" in checkpoint and "itos" in checkpoint:
            checkpoint["tokenizer_state"] = {
                "kind": "character",
                "stoi": checkpoint["stoi"],
                "itos": checkpoint["itos"],
            }
        else:
            checkpoint["tokenizer_state"] = {"kind": "tiktoken", "encoding_name": "gpt2"}

    return checkpoint
