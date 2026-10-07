import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:

        # computing forward
        z = np.dot(x,w) + b
        y_pred = 1/(1 + np.exp(-z))

        # loss
        L = 0.5 * (y_pred - y_true) ** 2

        # 2. Chain rule: dL/dz = (y_pred - y_true) * y_pred * (1 - y_pred)
        dL_dz = (y_pred - y_true) * y_pred * (1 - y_pred)

        # 3. Gradients w.r.t weights (dL/dw = dL/dz * x) and bias (dL/db = dL/dz)
        w_gradient = dL_dz * x
        b_gradient = dL_dz

        return (np.round(w_gradient,5),np.round(b_gradient,5))
