"""sft.py — supervised fine-tuning on instruction data with prompt masking."""

import json
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from model.gpt import GPT
import tiktoken

# ---- Config ----
BASE_CKPT = "checkpoints/ckpt_00018500.pt"
OUT_CKPT = "checkpoints/gpt_instruct.pt"
DATA_FILE = "dataset/alpaca.jsonl"
BATCH = 4
GRAD_ACCUM = 8
LR = 1e-4          # was 5e-5 — bump for faster convergence
MIN_LR = 1e-5      # was 5e-6
WARMUP_STEPS = 100
EPOCHS = 3         # was 2 — one more pass
LOG_EVERY = 20
EVAL_EVERY = 200
SEED = 1337

USER_TAG = "<|user|>"
ASSIST_TAG = "<|assistant|>"
END_TAG = "<|end|>"


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


class InstructDataset(Dataset):
    def __init__(self, path, enc, ctx):
        self.samples = []
        self.enc = enc
        self.ctx = ctx
        self.n_truncated = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        instruction = s.get("instruction", "").strip()
        inp = s.get("input", "").strip()
        user_content = f"{instruction}\n\n{inp}" if inp else instruction
        answer_content = s.get("output", "").strip()

        prompt = f"{USER_TAG}\n{user_content}\n{ASSIST_TAG}\n"
        answer = f"{answer_content}\n{END_TAG}\n"

        prompt_ids = self.enc.encode_ordinary(prompt)
        answer_ids = self.enc.encode_ordinary(answer)

        # --- KEY FIX: reserve room for at least some answer tokens ---
        # If prompt alone fills context, truncate the prompt so answer survives.
        max_prompt_len = max(1, self.ctx - min(len(answer_ids), self.ctx // 2))
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        # Truncate total to ctx
        input_ids = input_ids[: self.ctx]
        labels = labels[: self.ctx]

        # Pad to ctx
        pad_len = self.ctx - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def get_lr(step, total_steps, warmup, peak, floor):
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= total_steps:
        return floor
    progress = (step - warmup) / max(1, total_steps - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return floor + coeff * (peak - floor)


def compute_loss(logits, labels):
    """Autoregressive cross-entropy with -100 masking."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = compute_loss(logits, y)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    enc = get_tokenizer()

    # Load base model
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    CTX = cfg["context_length"]
    model = GPT(
        vocab_size=cfg["vocab_size"],
        context_length=CTX,
        model_dim=cfg["model_dim"],
        num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded base model from {BASE_CKPT} (step {ckpt.get('step','?')}), "
          f"ctx={CTX}, params={n_params/1e6:.2f}M")

    ds = InstructDataset(DATA_FILE, enc, CTX)

    # --- Sanity check: are labels actually being supervised? ---
    n_empty, n_short = 0, 0
    check_n = min(300, len(ds))
    for i in range(check_n):
        _, y = ds[i]
        n_sup = int((y != -100).sum().item())
        if n_sup == 0:
            n_empty += 1
        elif n_sup < 5:
            n_short += 1
    print(f"[sanity] checked {check_n} samples: "
          f"{n_empty} with NO supervised tokens, {n_short} with <5 tokens")
    if n_empty > check_n * 0.05:
        print("  ⚠️  WARNING: many samples have no answer tokens — "
              "increase ctx or shorten prompts.")

    n_val = max(200, len(ds) // 20)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"SFT dataset: {n_train} train, {n_val} val")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    model.train()

    steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
    total_opt_steps = steps_per_epoch * EPOCHS

    global_step = 0
    for epoch in range(EPOCHS):
        total = 0.0
        n = 0
        lr = LR
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = compute_loss(logits, y) / GRAD_ACCUM

            scaler.scale(loss).backward()
            total += loss.item() * GRAD_ACCUM
            n += 1

            is_last_micro = (step + 1) == len(train_loader)
            if (step + 1) % GRAD_ACCUM == 0 or is_last_micro:
                lr = get_lr(global_step, total_opt_steps, WARMUP_STEPS, LR, MIN_LR)
                for g in opt.param_groups:
                    g["lr"] = lr

                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % EVAL_EVERY == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  [eval] step {global_step} val_loss {val_loss:.4f} "
                          f"lr {lr:.2e}")

            if step % LOG_EVERY == 0:
                print(f"epoch {epoch+1} step {step} "
                      f"loss {total/max(n,1):.4f} lr {lr:.2e}")

        val_loss = evaluate(model, val_loader, device)
        print(f"=== epoch {epoch+1} done | train_loss {total/max(n,1):.4f} "
              f"| val_loss {val_loss:.4f} ===")

        torch.save({
            "model": model.state_dict(),
            "config": cfg,
            "sft": True,
            "epoch": epoch + 1,
        }, OUT_CKPT.replace(".pt", f"_epoch{epoch+1}.pt"))

    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "sft": True,
    }, OUT_CKPT)
    print(f"saved SFT checkpoint to {OUT_CKPT}")


if __name__ == "__main__":
    main()