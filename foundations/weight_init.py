import torch
import torch.nn as nn
import math


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> list[list[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> list[list[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> list[float]:
        torch.manual_seed(0)
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / dims[i])
            else:
                std = 1.0
            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)

        x = torch.randn(1, input_dim)
        stds = []
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds