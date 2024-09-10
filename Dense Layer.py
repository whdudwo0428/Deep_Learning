import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): #입력갯수 출력갯수
        self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        self.biases = np.random.uniform(0, 1, (1,n_neurons))

    def forward(self, inputs):
        layers_outputs = np.dot(inputs, np.array(self.weights)) + self.biases
        return layers_outputs

inputs,y = spiral_data(samples=100, classes=3)
plt.scatter(inputs[:, 0], inputs[:, 1], c=y, cmap = 'brg')
plt.show()

DNN = Layer_Dense(2,5) #위 샘플은 2차원공간에서 정의되기 때문에 인풋을2로 설정해야함
DNN2 = Layer_Dense(5,3)

outputs = DNN.forward(inputs)  # forward 메서드의 결과를 저장
print(outputs)  # 실제로 forward 메서드의 출력값을 출력



