import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        torch.manual_seed(0)
        ix = torch.randint(len(data) - context_length, (batch_size,))
        x = torch.stack([data[i:i + context_length] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix])
        return x, y