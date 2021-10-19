import numpy as np

X = inputs = [[1, 2, 3, 2.5],
             [2.0, 5.0, -1.0, 2.0],
             [-1.5, 2.7, 3.3, -0.8]]

'''
inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
outputs = []

#Basic algo for ReLU
for i in inputs:
    outputs.append(max(0, i))

print(outputs)
'''

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)
activation1 = Activation_ReLU()

#Forward the inputs
layer1.forward(X)
activation1.forward(layer1.output)

print(activation1.output)