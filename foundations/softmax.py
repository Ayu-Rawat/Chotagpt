import numpy as np
from numpy.typing import NDArray

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.exp(z - np.max(z))
        n = np.sum(z)
        return np.round(z/n, 4)
