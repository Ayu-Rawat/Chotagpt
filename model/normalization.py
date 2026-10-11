import numpy as np
from numpy.typing import NDArray
import math

class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        eps = 1e-5
        mean = np.mean(x)
        var = np.mean((x - mean)**2)

        x_hat = (x - mean)/np.sqrt(var + eps)
        out = gamma * x_hat + beta
        return np.round(out,5)
        
