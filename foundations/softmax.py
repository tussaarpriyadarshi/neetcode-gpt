import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        #prevent overflow ->subtract from max(z)
        #why suppose we have z=[1000,1001,1002] e^1000 becomes infinite,
        #numpy will throw error
        stable_z=z-np.max(z)
        exp_z=np.exp(stable_z)
        softmax_z=exp_z/np.sum(exp_z)
        return np.round(softmax_z,4)
        pass
