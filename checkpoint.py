"""Checkpoint persistence for trained GPT models."""

from pathlib import Path
from typing import Any, Mapping

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    stoi: Mapping[str, int],
    itos: Mapping[int, str],
    config: Mapping[str, Any],
) -> None:
    """Save model weights and the metadata required to reconstruct the model."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stoi": dict(stoi),
        "itos": dict(itos),
        "config": dict(config),
    }
    torch.save(checkpoint, Path(path))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint created by :func:`save_checkpoint` on the CPU."""
    checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} must contain a mapping")
    required_keys = {"model_state_dict", "stoi", "itos", "config"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Checkpoint at {path} is missing: {missing}")
    return checkpoint
