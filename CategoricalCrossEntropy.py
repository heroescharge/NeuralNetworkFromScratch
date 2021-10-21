import numpy as np


#Cross entropy formula

'''
softmax_output = [0.7, 0.1, 0.2]
target_output = [1,0,0]


loss = -1 * sum(target_output * np.log(softmax_output).T)
print(loss)
'''

#More complex implementation
'''
softmax_outputs = np.array([[0.7, 0.1, 0.2], 
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1] #First batch has correct answer with confidence 0.7, second batch has 0.5, third batch has 0.9

#Note you must account for 0, as the calculated error value will be log(0) which will crash; solve by using np.clip with some VERY small value
print(-np.log(softmax_outputs[[0, 1, 2], [class_targets]]))
'''


X = np.array([[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]])


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

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        #This is needed to check whether they pass one hot encoded values, or scalar class values
        if (len(y_true.shape) == 1):
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences) #Cross entropy error values
        return negative_log_likelihoods

layer1 = Layer_Dense(4, 5)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(5, 6)
activation2 = Activation_Softmax()

#Forward the inputs
layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

#Currently gives around 1/3 for everything bc of random distribution of inputs and weights
print(activation2.output)

#Correct class values for activation2
y = np.array([0, 1, 3])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print(loss)