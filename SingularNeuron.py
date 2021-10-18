import numpy as np

#Parameters for 3 neurons input and 3 neurons output

#Inputs and weights from previous neuron
inputs = [1.5, 2.1, 3.4]
weights = [[1, 2.1, 3.1], [1, 2, 3], [2, 4, 5]]
#Bias for current neuron
biases = [2, 3, 4]

#make sure weights, inputs NOT other way around
outputs = np.dot(weights, inputs) + biases

print(outputs)
