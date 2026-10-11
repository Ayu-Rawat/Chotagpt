import torch, tiktoken
from model.gpt import GPT

enc = tiktoken.get_encoding("gpt2")

# --- Test the BASE model first (not SFT) ---
ckpt = torch.load("checkpoints/ckpt_00018500.pt", map_location="cpu", weights_only=False)
cfg = ckpt["config"]
m = GPT(vocab_size=cfg["vocab_size"], context_length=cfg["context_length"],
        model_dim=cfg["model_dim"], num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"])
m.load_state_dict(ckpt["model"]); m.eval()
print(f"BASE params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M, ctx={cfg['context_length']}")

prompt = "The sky is blue because"
ids = torch.tensor([enc.encode_ordinary(prompt)])
with torch.no_grad():
    out = m.generate(ids, max_new_tokens=50) if hasattr(m, "generate") else None

# manual generate
gen = ids.tolist()[0]
with torch.no_grad():
    for _ in range(50):
        inp = torch.tensor([gen[-cfg["context_length"]:]])
        logits = m(inp)[0, -1]
        nxt = int(torch.argmax(logits))
        gen.append(nxt)
print("BASE continuation:", repr(enc.decode(gen)))