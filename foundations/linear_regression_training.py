import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        rows = len(X)
        cols = len(X[0])
        w = np.zeros(cols)
        learning_rate = 0.01

        for i in range(num_iterations):
            # computing y pred
            y_pred = self.get_model_prediction(X,initial_weights)

            # compute gradient
            for j in range(cols):
                w[j] = self.get_derivative(y_pred,Y,rows,X,j)

            # update weights
            for j in range(cols):
                initial_weights[j] -= learning_rate * w[j]


        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        return np.round(initial_weights, 5)
