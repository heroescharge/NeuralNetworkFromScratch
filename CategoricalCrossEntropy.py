import numpy as np

softmax_output = [0.7, 0.1, 0.2]
target_output = [1,0,0]

#Cross entropy formula
loss = -1 * sum(target_output * np.log(softmax_output).T)

print(loss)