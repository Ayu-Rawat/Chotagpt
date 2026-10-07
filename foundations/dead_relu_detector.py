import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.ReLU):
                    # A neuron is dead if it outputs 0 for ALL samples in the batch
                    dead = (x == 0).all(dim=0).float().mean().item()
                    dead_fractions.append(round(dead, 4))
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if len(dead_fractions) == 0:
            return 'healthy'

        max_frac = max(dead_fractions)

        # Any layer > 0.5 dead -> use LeakyReLU
        if max_frac > 0.5:
            return 'use_leaky_relu'

        # First layer > 0.3 dead -> reinitialize weights
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        # Dead fraction increases with depth -> reduce learning rate
        if len(dead_fractions) >= 2:
            increasing = all(
                dead_fractions[i] < dead_fractions[i + 1]
                for i in range(len(dead_fractions) - 1)
            )
            if increasing and dead_fractions[-1] > 0.1:
                return 'reduce_learning_rate'

        # All layers < 0.1 dead -> healthy
        if max_frac < 0.1:
            return 'healthy'

        return 'healthy'