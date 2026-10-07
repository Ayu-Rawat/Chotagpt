import torch
import torch.nn as nn
import torch.nn.functional as F

class Solution:
    def train(
        self,
        model: nn.Module,
        data: torch.Tensor,
        epochs: int,
        context_length: int,
        batch_size: int,
        lr: float,
        device: torch.device | None = None,
    ) -> float:
        device = device or next(model.parameters()).device
        model.to(device)
        data = data.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        model.train()

        for _ in range(epochs):
            ix = torch.randint(
                len(data) - context_length, (batch_size,), device=device
            )
            x = torch.stack([data[i:i + context_length] for i in ix])
            y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix])

            with torch.amp.autocast(
                device_type="cuda", enabled=device.type == "cuda"
            ):
                logits = model(x)
                B, T, C = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, C), y.reshape(B * T))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        return loss.item()