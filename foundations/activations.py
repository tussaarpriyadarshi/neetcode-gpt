import numpy as np
from numpy.typing import NDArray

"""a neuron usually does calculate the raw score i.e, z
z=wx+b why do we need raw score ,think of it we score marks in exam
that marks helps to know rank,divison,pass/fail
output is activation(z)
now this is passed through sigmoid or relu
because the raw score we calculated ,z is just a linear function
no matter how many linear function we put in layer ,we only get linear function
we need non linearity ,which will be provided by relu or sigmoid
why do we need non linearity:
uppose you want a model to classify this:
points inside a circle → class A
points outside → class B
A straight line cannot separate them.
You need curved decision boundaries.
Non-linear activations let transformers build complex semantic relationships across layers.
"""


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        answer=1/(1+np.exp(-z))
        return np.round(answer,5)
        pass

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        ans=np.maximum(0,z)
        return np.round(ans,5)
        pass
