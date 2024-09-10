import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8],
          [1.3, 0.6, -3.2, 1.3],
          [1.2, -0.1, 0.2, 0.7]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

layers_outputs = np.dot(inputs,np.array(weights).T)+biases
print(layers_outputs)

weights2 = [[0.3, 0.6, -0.9],
           [0.4, -0.92, 0.26],
           [-0.22, -0.25, 0.57]]
biases2 = [1.0, -1.5, -0.5]

layers_outputs2 = np.dot(layers_outputs,np.array(weights2).T)+biases
print(layers_outputs2)
