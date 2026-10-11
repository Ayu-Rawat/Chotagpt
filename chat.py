"""chat.py — interactive chat with the SFT model."""

import torch
import tiktoken
from model.gpt import GPT

CKPT = "checkpoints/gpt_instruct.pt"
USER_TAG = "<|user|>"
ASSIST_TAG = "<|assistant|>"
END_TAG = "<|end|>"
MAX_NEW = 200
TEMPERATURE = 0.8
TOP_K = 40


@torch.no_grad()
def generate(model, enc, history, ctx, device,
             max_new=MAX_NEW, temp=TEMPERATURE, top_k=TOP_K):
    prompt = ""
    for user, assistant in history:
        prompt += f"{USER_TAG}\n{user}\n{ASSIST_TAG}\n"
        if assistant is not None:
            prompt += f"{assistant}\n{END_TAG}\n"

    # --- KEY FIX: only use the LAST turn's prompt to avoid old context issues,
    #     but also print raw for debugging. ---
    ids = enc.encode_ordinary(prompt)
    # Keep the prompt within ctx (leave room for new tokens)
    if len(ids) > ctx - max_new:
        ids = ids[-(ctx - max_new):]
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    end_ids = enc.encode_ordinary(END_TAG)
    eos_id = enc.eot_token  # 50256 for gpt2

    generated = []
    for _ in range(max_new):
        inp = ids[:, -ctx:]
        logits = model(inp)[0, -1].float()
        logits = logits / max(temp, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()

        # Stop on GPT-2 EOS too
        if nxt == eos_id:
            break

        generated.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)

        # Stop on <|end|>
        if len(generated) >= len(end_ids):
            if generated[-len(end_ids):] == end_ids:
                generated = generated[:-len(end_ids)]
                break

    return enc.decode(generated).strip()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding("gpt2")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
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
    print(f"loaded {CKPT}. type 'exit' to quit.\n")

    # --- Debug helper: print raw tokens once ---
    debug = True

    history = []
    while True:
        try:
            user = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in ("exit", "quit"):
            break
        if not user:
            continue
        history.append([user, None])

        if debug:
            # show what the raw prompt looks like
            raw = ""
            for u, a in history:
                raw += f"{USER_TAG}\n{u}\n{ASSIST_TAG}\n"
                if a is not None:
                    raw += f"{a}\n{END_TAG}\n"
            print(f"[debug] raw prompt: {raw!r}")
            debug = False

        reply = generate(model, enc, history, cfg["context_length"], device)
        if not reply:
            reply = "(empty — model produced no tokens)"
        history[-1][1] = reply
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()