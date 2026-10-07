import torch
import torch.nn as nn
import torch.nn.functional as F

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            torch.manual_seed(epoch)
            ix = torch.randint(len(data) - context_length, (batch_size,))
            x = torch.stack([data[i:i + context_length] for i in ix])
            y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix])

            logits = model(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)