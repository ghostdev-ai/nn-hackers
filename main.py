import numpy as np

class Linear:
    def __init__(self, 
                 n_inputs: int, 
                 n_neurons: int
                 ) -> None:
        # Initialize weights and biases

        # np.random.randn -> produces a Gaussian distribution with a mean 0 and variance 1. Meaning
        # random numbers will be generated, positive and negative, centered at o with a mean close to 0.
        # Note: neural networks work best with values between -1 and 1.
        
        # Multiplying by 0.1 generates numbers that are couple of magnitudes smaller. Otherwise,
        # model will take longer to fit the data as starting values will be disproportionatley large compared 
        # to updates during training. 

        # Main idea: start a model with non-zero values small enough they won't affect training. 
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # initialized as row vector, allows us to add to the results of the dot product without transposing the matrix. 
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, 
                x: np.ndarray
                ) -> np.ndarray:
        # Calculate output values from x (i.e., inputs)
        self.output = np.dot(x, self.weights) + self.biases


class ReLU:
    
    # Forward pass
    def forward(self, 
                x: np.ndarray
                ) -> np.ndarray:
        # Calculate output values from x (i.e., inputs)
        self.output = np.maximum(0, x)