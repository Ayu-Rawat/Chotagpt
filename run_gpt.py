"""Train, checkpoint, reload, and sample from the repository's GPT model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from checkpoint import load_checkpoint, save_checkpoint
from data.vocab import Solution as Vocabulary
from model.gpt import GPT


DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "A small language model learns to predict the next character. "
)


def load_text(path: Path) -> str:
    """Read a local dataset, falling back to a useful sample when absent."""
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if text:
            return text
    return DEFAULT_TEXT


def sample_batch(
    data: torch.Tensor, context_length: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random input and next-token target windows."""
    if data.ndim != 1 or len(data) <= context_length:
        raise ValueError("Dataset must be a 1D tensor longer than context_length")
    starts = torch.randint(
        len(data) - context_length, (batch_size,), device=data.device
    )
    inputs = torch.stack([data[start : start + context_length] for start in starts])
    targets = torch.stack(
        [data[start + 1 : start + 1 + context_length] for start in starts]
    )
    return inputs, targets


def train_model(
    model: GPT,
    data: torch.Tensor,
    epochs: int,
    context_length: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    """Train the model and print loss progress after every epoch."""
    model.to(device)
    data = data.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    for epoch in range(1, epochs + 1):
        inputs, targets = sample_batch(data, context_length, batch_size)
        inputs = inputs.to(device, non_blocking=use_amp)
        targets = targets.to(device, non_blocking=use_amp)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(inputs)
            batch_size_actual, sequence_length, vocabulary_size = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch_size_actual * sequence_length, vocabulary_size),
                targets.reshape(batch_size_actual * sequence_length),
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            if use_amp:
                allocated_mb = torch.cuda.memory_allocated(device) / 1e6
                memory = f" - VRAM: {allocated_mb:.1f} MB"
            else:
                memory = ""
            print(
                f"epoch {epoch}/{epochs} - loss: {loss.item():.6f}{memory}",
                flush=True,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("input.txt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("gpt_checkpoint.pt"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--new-tokens", type=int, default=100)
    parser.add_argument("--prompt", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Training on CUDA: {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print("Training on CPU", flush=True)
    text = load_text(args.input)
    vocabulary = Vocabulary()
    stoi, itos = vocabulary.build_vocab(text)
    encoded = torch.tensor(vocabulary.encode(text, stoi), dtype=torch.long)
    if len(encoded) <= args.context_length:
        repeats = args.context_length // len(encoded) + 1
        text = text * repeats
        stoi, itos = vocabulary.build_vocab(text)
        encoded = torch.tensor(vocabulary.encode(text, stoi), dtype=torch.long)

    config: dict[str, Any] = {
        "vocab_size": len(stoi),
        "context_length": args.context_length,
        "model_dim": args.model_dim,
        "num_blocks": args.num_blocks,
        "num_heads": args.num_heads,
    }
    model = GPT(**config)
    train_model(
        model,
        encoded,
        args.epochs,
        args.context_length,
        args.batch_size,
        args.learning_rate,
        device,
    )
    save_checkpoint(args.checkpoint, model, stoi, itos, config)
    print(f"saved checkpoint to {args.checkpoint}")

    checkpoint = load_checkpoint(args.checkpoint)
    reloaded_model = GPT(**checkpoint["config"]).to(device)
    reloaded_model.load_state_dict(checkpoint["model_state_dict"])
    reloaded_model.eval()
    prompt = args.prompt or text[:1]
    prompt_ids = [checkpoint["stoi"][character] for character in prompt]
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        from generate import Solution as Generator

        generated = Generator().generate(
            reloaded_model,
            args.new_tokens,
            context,
            checkpoint["config"]["context_length"],
            checkpoint["itos"],
        )
    print(f"generated: {prompt}{generated}")


if __name__ == "__main__":
    main()
