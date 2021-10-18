import numpy as np

np.random.seed(0)

#4 inputs per batch, 3 batches
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#4 inputs from above, we CHOOSE 5 neurons on output
layer1 = Layer_Dense(4, 5)
#5 inputs because layer1 has 5 outputs, we CHOOSE 2 neurouns on output
layer2 = Layer_Dense(5, 2)

#Do computation
layer1.forward(X)
layer2.forward(layer1.output)
#Number of batches x number of neurons matrix
print(layer1.output)
print(layer2.output)