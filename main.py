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


class Softmax:
    def forward(self, 
                x: np.ndarray
                ) -> np.ndarray:
        expValues = np.exp(x - np.max(x, axis=1, keepdims=True))
        normValues = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = normValues

class Loss:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)
        return data_loss
    

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correctConfidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correctConfidences = np.sum(y_pred_clipped * y_true, axis=1)

        print(correctConfidences)
        negativeLogLiklihoods = -np.log(correctConfidences)
        return negativeLogLiklihoods



inputs = np.random.randn(5, 2)
linear1 = Linear(2, 3)
relu1 = ReLU()
linear2 = Linear(3, 3)
softmax1 = Softmax()

linear1.forward(inputs)
relu1.forward(linear1.output)
linear2.forward(relu1.output)
softmax1.forward(linear2.output)

print(softmax1.output)

lossFunc = CategoricalCrossEntropy()
loss = lossFunc.calculate(softmax1.output, np.array([0, 1, 1, 0, 2]))

print(f"Loss: {loss}")