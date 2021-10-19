import math
import numpy as np

#Basic implementation of softmax
''' 
layer_outputs = [4.8, 1.21, 2.385]

#Basic exponentiation of each value in array 
exp_values = np.exp(layer_outputs)

print(exp_values)

#Normalize outputs
norm_values = exp_values/np.sum(exp_values)

print(norm_values)
print(sum(norm_values))
'''

#Complex implementation of softmax

'''
layer_outputs = [[4.8, 1.21, 2.385],
                 [4.2, 8.1, -9.3],
                 [4.5, -0.3, 1]]

#Basic exponentiation of each value in array 
exp_values = np.exp(layer_outputs)

#Get matrix with sum of the rows
#keepdims keeps it in a "tranposed" for easy multiplicaiton
norm_values = exp_values/np.sum(layer_outputs, axis=1, keepdims = True)

print(norm_values)
'''

#Class implementation of activiation function

X = inputs = [[1, 2, 3, 2.5],
             [2.0, 5.0, -1.0, 2.0],
             [-1.5, 2.7, 3.3, -0.8]]

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

layer1 = Layer_Dense(4, 5)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(5, 3)
activation2 = Activation_Softmax()

#Forward the inputs
layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

#Currently gives around 1/3 for everything bc of random distribution of inputs and weights
print(activation2.output)