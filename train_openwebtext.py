"""Streamed OpenWebText pretraining for the repository GPT model.

This script streams OpenWebText directly from Hugging Face instead of
downloading the full corpus locally. It is intended for larger runs than the
smaller chunked trainer in run_gpt.py.

Usage:
    python train_openwebtext.py
    python train_openwebtext.py --resume
    python train_openwebtext.py --resume checkpoints/ckpt_0005000.pt
"""

import argparse
import math
import os
import signal
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

# ---- your model ----
from model.gpt import GPT


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    # model
    vocab_size: int = 50257          # GPT-2 BPE
    context_length: int = 256        # keep small for 8GB VRAM
    num_blocks: int = 6
    num_heads: int = 6
    model_dim: int = 384

    # data
    dataset_name: str = "Skylion007/openwebtext"
    dataset_config: str = None
    text_field: str = "text"

    # optimization
    micro_batch_size: int = 8        # per forward pass
    grad_accum_steps: int = 32       # effective batch = 8 * 32 = 256 sequences
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # training length
    max_steps: int = 200_000         # ~ how long you want to train

    # checkpointing
    out_dir: str = "checkpoints"
    ckpt_every_steps: int = 500
    ckpt_every_minutes: int = 30
    keep_last_n: int = 3             # rotate older checkpoints

    # misc
    log_every: int = 10
    seed: int = 1337
    grad_checkpointing: bool = True
    compile: bool = False            # torch.compile (optional, slow first step)


# -----------------------------
# Tokenizer metadata
# -----------------------------
def get_tokenizer_meta():
    """Return only picklable primitives (eot token id, vocab size).

    The actual encoder is created lazily inside the dataset workers to avoid
    pickling closures across process boundaries.
    """
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc.eot_token, enc.n_vocab


# -----------------------------
# Streaming dataset
# -----------------------------
class StreamedTokens(IterableDataset):
    """
    Yields fixed-length (input, target) blocks of token IDs from a streamed
    HF dataset. The tokenizer is constructed lazily inside __iter__ so the
    dataset object stays picklable (important on Windows / spawn start method).
    """

    def __init__(self, eot_token, context_length,
                 dataset_name, dataset_config, text_field,
                 start_skip_docs: int = 0, seed: int = 1337):
        super().__init__()
        # store only primitives — no functions, no encoders, no closures
        self.eot = int(eot_token)
        self.ctx = int(context_length)
        self.dataset_name = str(dataset_name)
        self.dataset_config = dataset_config
        self.text_field = str(text_field)
        self.start_skip_docs = int(start_skip_docs)
        self.seed = int(seed)

    def _stream(self, worker_id: int = 0):
        # Create tiktoken encoder inside the worker. Cheap, and keeps
        # the dataset object picklable.
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)

        # Only shard if we actually have multiple workers.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_id)

        # Approximate resume: skip already-seen docs on worker 0 only.
        if self.start_skip_docs > 0 and worker_id == 0:
            ds = ds.skip(self.start_skip_docs)

        buf = []
        docs_seen = 0
        for ex in ds:
            text = ex.get(self.text_field, "")
            if not text:
                continue
            ids = enc.encode_ordinary(text)
            ids.append(self.eot)
            buf.extend(ids)
            docs_seen += 1
            while len(buf) >= self.ctx + 1:
                chunk = buf[: self.ctx + 1]
                del buf[: self.ctx]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y, docs_seen

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info else 0
        yield from self._stream(wid)


# -----------------------------
# LR schedule: linear warmup + cosine decay
# -----------------------------
def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_checkpoint(path: Path, model, optimizer, step, cfg: TrainConfig,
                    rng_cpu, rng_cuda, docs_seen: int, tokens_seen: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    # Always store RNG states as CPU uint8 tensors, detached from the graph.
    if rng_cpu is not None:
        rng_cpu = rng_cpu.detach().to("cpu", dtype=torch.uint8)
    if rng_cuda is not None:
        rng_cuda = rng_cuda.detach().to("cpu", dtype=torch.uint8)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": asdict(cfg),
        "rng_cpu": rng_cpu,
        "rng_cuda": rng_cuda,
        "docs_seen": docs_seen,
        "tokens_seen": tokens_seen,
    }
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic


def load_checkpoint(path: Path, model, optimizer, device):
    print(f"[resume] loading {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    # --- RNG restore (must be CPU ByteTensor) ---
    rng_cpu = ckpt.get("rng_cpu")
    if rng_cpu is not None:
        if not isinstance(rng_cpu, torch.Tensor):
            rng_cpu = torch.tensor(rng_cpu, dtype=torch.uint8)
        rng_cpu = rng_cpu.detach().to("cpu", dtype=torch.uint8)
        torch.set_rng_state(rng_cpu)

    rng_cuda = ckpt.get("rng_cuda")
    if rng_cuda is not None and torch.cuda.is_available():
        if not isinstance(rng_cuda, torch.Tensor):
            rng_cuda = torch.tensor(rng_cuda, dtype=torch.uint8)
        rng_cuda = rng_cuda.detach().to("cpu", dtype=torch.uint8)
        torch.cuda.set_rng_state(rng_cuda)

    return ckpt


def latest_checkpoint(out_dir: Path):
    ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


def rotate_checkpoints(out_dir: Path, keep_last_n: int):
    ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    for old in ckpts[:-keep_last_n]:
        try:
            old.unlink()
            print(f"[ckpt] removed old {old.name}")
        except OSError:
            pass


# -----------------------------
# Training
# -----------------------------
def train(cfg: TrainConfig, resume_path: str | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    eot, vocab_size = get_tokenizer_meta()
    cfg.vocab_size = vocab_size
    print(f"[tokenizer] vocab_size={vocab_size}")

    # model
    model = GPT(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        model_dim=cfg.model_dim,
        num_blocks=cfg.num_blocks,
        num_heads=cfg.num_heads,
    ).to(device)

    # gradient checkpointing hook (if your GPT has blocks with .forward)
    if cfg.grad_checkpointing and hasattr(model, "transformer_blocks"):
        try:
            from torch.utils.checkpoint import checkpoint as _ckpt
            for blk in model.transformer_blocks:
                orig = blk.forward

                def make(orig_fn):
                    def fwd(*args, **kwargs):
                        return _ckpt(orig_fn, *args, use_reentrant=False, **kwargs)
                    return fwd

                blk.forward = make(orig)
            print("[opt] gradient checkpointing enabled")
        except Exception as e:
            print(f"[warn] could not enable grad checkpointing: {e}")

    if cfg.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[opt] torch.compile enabled")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    start_step = 0
    docs_seen = 0
    tokens_seen = 0
    best_loss = float("inf")

    if resume_path:
        p = Path(resume_path)
        if p.is_dir():
            p = latest_checkpoint(p)
        if p and p.exists():
            ckpt = load_checkpoint(p, model, optimizer, device)
            start_step = ckpt.get("step", 0)
            docs_seen = ckpt.get("docs_seen", 0)
            tokens_seen = ckpt.get("tokens_seen", 0)
            best_loss = ckpt.get("best_loss", float("inf"))
            print(f"[resume] step={start_step} docs_seen={docs_seen} tokens_seen={tokens_seen}")
        else:
            print(f"[resume] no checkpoint found at {resume_path}, starting fresh")

    # Streaming datasets don't benefit from workers (I/O bound) and
    # workers break on Windows/spawn with HF streaming. Force 0.
    num_workers = 0

    dataset = StreamedTokens(
        eot_token=eot,
        context_length=cfg.context_length,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        text_field=cfg.text_field,
        start_skip_docs=docs_seen,
        seed=cfg.seed + start_step,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=None,   # only valid when num_workers > 0
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    model.train()

    # graceful interrupt
    stop_flag = {"stop": False}

    def _handler(sig, frame):
        print("\n[interrupt] will save and exit after current step...")
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_time = time.time()

    data_iter = iter(loader)
    step = start_step
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    while step < cfg.max_steps and not stop_flag["stop"]:
        # set LR for this step
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        micro_losses = []

        for micro in range(cfg.grad_accum_steps):
            try:
                x, y, docs = next(data_iter)
            except StopIteration:
                print("[data] exhausted stream, restarting...")
                data_iter = iter(loader)
                x, y, docs = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            micro_losses.append(loss.item() * cfg.grad_accum_steps)
            docs_seen = max(docs_seen, int(docs.max().item()) if hasattr(docs, "max") else docs_seen)

        # clip + step
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_seen += cfg.micro_batch_size * cfg.grad_accum_steps * cfg.context_length
        step_loss = sum(micro_losses) / len(micro_losses)
        running_loss += step_loss
        running_count += 1

        # log
        if step % cfg.log_every == 0:
            avg = running_loss / running_count
            dt = time.time() - t0
            tok_s = (cfg.micro_batch_size * cfg.grad_accum_steps *
                     cfg.context_length * running_count) / max(dt, 1e-6)
            vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            print(f"step {step:>7} | loss {avg:.4f} | lr {lr:.2e} | "
                  f"{tok_s:,.0f} tok/s | {vram:.2f} GB | docs {docs_seen:,}")
            running_loss = 0.0
            running_count = 0
            t0 = time.time()
            if avg < best_loss:
                best_loss = avg

        # periodic checkpoint by step
        do_ckpt = (step % cfg.ckpt_every_steps == 0)
        # periodic checkpoint by time
        if (time.time() - last_ckpt_time) > cfg.ckpt_every_minutes * 60:
            do_ckpt = True

        if do_ckpt or stop_flag["stop"]:
            ckpt_path = out_dir / f"ckpt_{step:08d}.pt"
            save_checkpoint(
                ckpt_path, model, optimizer, step, cfg,
                torch.get_rng_state(),
                torch.cuda.get_rng_state() if device.type == "cuda" else None,
                docs_seen, tokens_seen,
            )
            # also keep a "latest.pt" copy for easy resume
            latest = out_dir / "latest.pt"
            save_checkpoint(
                latest, model, optimizer, step, cfg,
                torch.get_rng_state(),
                torch.cuda.get_rng_state() if device.type == "cuda" else None,
                docs_seen, tokens_seen,
            )
            rotate_checkpoints(out_dir, cfg.keep_last_n)
            last_ckpt_time = time.time()
            print(f"[ckpt] saved {ckpt_path.name} (step={step})")

        if stop_flag["stop"]:
            print("[interrupt] clean exit")
            break

    # final save
    final = out_dir / f"ckpt_final_{step:08d}.pt"
    save_checkpoint(
        final, model, optimizer, step, cfg,
        torch.get_rng_state(),
        torch.cuda.get_rng_state() if device.type == "cuda" else None,
        docs_seen, tokens_seen,
    )
    print(f"[done] final checkpoint: {final}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", nargs="?", const="checkpoints/latest.pt", default=None,
                   help="Path to checkpoint file or directory. Use --resume with no value "
                        "to resume from checkpoints/latest.pt")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--micro_batch_size", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ckpt_every_steps", type=int, default=None)
    p.add_argument("--ckpt_every_minutes", type=int, default=None)
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig()
    cfg.out_dir = args.out_dir
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.micro_batch_size is not None:
        cfg.micro_batch_size = args.micro_batch_size
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = args.grad_accum_steps
    if args.context_length is not None:
        cfg.context_length = args.context_length
    if args.lr is not None:
        cfg.lr = args.lr
    if args.ckpt_every_steps is not None:
        cfg.ckpt_every_steps = args.ckpt_every_steps
    if args.ckpt_every_minutes is not None:
        cfg.ckpt_every_minutes = args.ckpt_every_minutes
    if args.no_grad_ckpt:
        cfg.grad_checkpointing = False
    if args.compile:
        cfg.compile = True

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()