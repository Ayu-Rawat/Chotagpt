"""Checkpoint inspector and sampler for the streamed OpenWebText GPT.

Usage:
    python check_checkpoint.py                          # inspect checkpoints/latest.pt
    python check_checkpoint.py checkpoints/ckpt_0002000.pt
    python check_checkpoint.py --list                   # list all checkpoints
    python check_checkpoint.py --prompt "The quick brown fox"
    python check_checkpoint.py --sample --n 80 --temp 0.8 --top_k 40
    python check_checkpoint.py --hist                   # loss history from logs/
    python check_checkpoint.py --all                    # run every check
"""

import argparse
import glob
import math
import re
from pathlib import Path

import torch

# ---- your model ----
from model.gpt import GPT


# -----------------------------
# Helpers
# -----------------------------
def human(n: int) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n}"
        n /= 1000
    return f"{n:.1f}P"


def find_latest(out_dir: str = "checkpoints") -> Path | None:
    d = Path(out_dir)
    latest = d / "latest.pt"
    if latest.exists():
        return latest
    ckpts = sorted(d.glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


def list_checkpoints(out_dir: str = "checkpoints"):
    d = Path(out_dir)
    ckpts = sorted(d.glob("ckpt_*.pt"))
    if not ckpts:
        print(f"no checkpoints in {d}/")
        return
    print(f"{'file':<32} {'size':>10}  {'step':>10}  {'tokens_seen':>16}")
    print("-" * 74)
    for c in ckpts:
        size_mb = c.stat().st_size / 1e6
        try:
            ckpt = torch.load(c, map_location="cpu", weights_only=False)
            step = ckpt.get("step", "?")
            tokens = ckpt.get("tokens_seen", 0)
            print(f"{c.name:<32} {size_mb:>8.1f}MB  {str(step):>10}  {human(tokens):>16}")
        except Exception as e:
            print(f"{c.name:<32} {size_mb:>8.1f}MB  [unreadable: {e}]")
    latest = d / "latest.pt"
    if latest.exists():
        print(f"\nlatest.pt -> points to step content inside it (see above)")


# -----------------------------
# Inspect
# -----------------------------
def inspect(path: Path, cfg_override: dict | None = None):
    print(f"\n=== inspecting {path} ===")
    size_mb = path.stat().st_size / 1e6
    print(f"file size:     {size_mb:.1f} MB")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    step = ckpt.get("step", 0)
    cfg = ckpt.get("config", {})
    tokens_seen = ckpt.get("tokens_seen", 0)
    docs_seen = ckpt.get("docs_seen", 0)

    print(f"step:          {step:,}")
    print(f"docs_seen:     {docs_seen:,}")
    print(f"tokens_seen:   {tokens_seen:,}  ({human(tokens_seen)})")

    # ---- token math check ----
    micro = cfg.get("micro_batch_size")
    accum = cfg.get("grad_accum_steps")
    ctx = cfg.get("context_length")
    if micro and accum and ctx:
        expected = step * micro * accum * ctx
        ok = "OK" if expected == tokens_seen else "MISMATCH"
        print(f"expected tok:  {expected:,}  ({human(expected)})  [{ok}]")
        if expected != tokens_seen:
            print(f"  WARNING: tokens_seen does not match step * micro * accum * ctx")
            print(f"           micro={micro} accum={accum} ctx={ctx} step={step}")

    # ---- config ----
    print(f"config:")
    for k in sorted(cfg.keys()):
        print(f"  {k}: {cfg[k]}")

    # ---- optimizer state ----
    opt = ckpt.get("optimizer", {})
    opt_state = opt.get("state", {})
    opt_has_state = len(opt_state) > 0
    print(f"optimizer state tensors: {len(opt_state)}  "
          f"[{'OK' if opt_has_state else 'MISSING — resume will lose momentum'}]")

    # ---- model tensors ----
    model_sd = ckpt.get("model", {})
    n_params = sum(t.numel() for t in model_sd.values() if hasattr(t, "numel"))
    print(f"model tensors:  {len(model_sd)}")
    print(f"model params:   {n_params:,}  ({human(n_params)})")

    # ---- RNG ----
    print(f"rng_cpu present:  {ckpt.get('rng_cpu') is not None}")
    print(f"rng_cuda present: {ckpt.get('rng_cuda') is not None}")

    return ckpt


# -----------------------------
# Load model from checkpoint
# -----------------------------
def build_model_from_ckpt(ckpt, device):
    cfg = ckpt["config"]
    model = GPT(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        model_dim=cfg["model_dim"],
        num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


# -----------------------------
# Sampling
# -----------------------------
@torch.no_grad()
def generate(model, cfg, prompt: str, n_tokens: int, temperature: float,
             top_k: int | None, device, greedy: bool):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    ids = enc.encode_ordinary(prompt) if prompt else [enc.eot_token]
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    ctx = cfg["context_length"]

    for _ in range(n_tokens):
        inp = ids[:, -ctx:]
        logits = model(inp)[0, -1]  # (vocab,)

        if greedy:
            nxt = torch.argmax(logits).item()
        else:
            logits = logits / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()

        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)

    return enc.decode(ids[0].tolist())


# -----------------------------
# Loss history from logs
# -----------------------------
def show_loss_history(log_dir: str = "logs", tail: int = 60):
    files = sorted(glob.glob(f"{log_dir}/*.log"))
    if not files:
        print(f"no log files in {log_dir}/ — did you run with `| tee logs/...` ?")
        return
    print(f"found {len(files)} log file(s), showing last {tail} loss lines from {files[-1]}")
    print()
    pat = re.compile(r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)")
    rows = []
    for f in files:
        try:
            with open(f, "r", errors="replace") as fh:
                for line in fh:
                    m = pat.search(line)
                    if m:
                        rows.append((int(m.group(1)), float(m.group(2))))
        except OSError:
            continue
    rows.sort()
    if not rows:
        print("no loss lines found")
        return
    print(f"{'step':>8}  {'loss':>8}")
    print("-" * 18)
    for s, l in rows[-tail:]:
        print(f"{s:>8}  {l:>8.4f}")

    # crude trend: fit slope over the last 10 points
    if len(rows) >= 10:
        last10 = rows[-10:]
        dstep = last10[-1][0] - last10[0][0]
        dloss = last10[-1][1] - last10[0][1]
        print()
        print(f"last 10 logs: Δstep={dstep}, Δloss={dloss:+.4f} "
              f"({'decreasing — good' if dloss < 0 else 'FLAT OR RISING — check LR / data'})")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", default=None,
                   help="path to checkpoint .pt (default: checkpoints/latest.pt)")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--list", action="store_true", help="list all checkpoints and exit")
    p.add_argument("--inspect", action="store_true", help="inspect the checkpoint (default action)")
    p.add_argument("--sample", action="store_true", help="generate text from the checkpoint")
    p.add_argument("--prompt", default="The quick brown fox", help="prompt for --sample")
    p.add_argument("--n", type=int, default=60, help="number of tokens to generate")
    p.add_argument("--temp", type=float, default=0.8, help="sampling temperature")
    p.add_argument("--top_k", type=int, default=40, help="top-k filter (0 = off)")
    p.add_argument("--greedy", action="store_true", help="greedy (argmax) sampling")
    p.add_argument("--hist", action="store_true", help="show loss history from logs/")
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--hist_tail", type=int, default=60)
    p.add_argument("--all", action="store_true", help="run list + inspect + sample + hist")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    run_list = args.list or args.all
    run_inspect = args.inspect or args.all or (not args.list and not args.sample
                                               and not args.hist and not args.all)
    run_sample = args.sample or args.all
    run_hist = args.hist or args.all

    if run_list:
        list_checkpoints(args.out_dir)

    if run_inspect or run_sample:
        path = Path(args.ckpt) if args.ckpt else find_latest(args.out_dir)
        if path is None or not path.exists():
            print(f"no checkpoint found (looked at {args.ckpt or args.out_dir + '/latest.pt'})")
            return
        ckpt = inspect(path) if run_inspect else torch.load(path, map_location="cpu", weights_only=False)

        if run_sample:
            print(f"\n=== sampling on {device} ===")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model, cfg = build_model_from_ckpt(ckpt, device)
            print(f"prompt: {args.prompt!r}")
            print(f"sampling: n={args.n} temp={args.temp} top_k={args.top_k} greedy={args.greedy}")
            print("-" * 60)
            out = generate(model, cfg, args.prompt, args.n, args.temp,
                           None if args.top_k == 0 else args.top_k, device, args.greedy)
            print(out)
            print("-" * 60)

    if run_hist:
        print()
        show_loss_history(args.log_dir, args.hist_tail)


if __name__ == "__main__":
    main()