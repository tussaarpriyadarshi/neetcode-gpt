import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsil=1e-7
        term1=y_true*np.log(y_pred+epsil)
        term2=(1-y_true)*np.log(1-y_pred+epsil)
        loss=-np.mean(term1+term2)
        return np.round(loss,4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon=1e-7
        log_preds=y_true*np.log(y_pred+epsilon)
        total_sum=np.sum(log_preds)
        n_samples=y_true.shape[0]
        loss=-(total_sum/n_samples)
        return np.round(loss,4)
        pass
