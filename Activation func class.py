import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

nnfs.init()  # 랜덤시드들이 고정됨 /이 기준으로만 랜덤값이 설정됨


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialize_method='random'):  # 입력갯수 출력갯수
        match initialize_method:  # 가중치 초기화 방법 선택       #인자에서  'random'을 미리 대입시켜두면 함수를 부를 때 기본값으로 들어감
            case 'uniform':
                self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
            case 'xavier':
                self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)  # Xavier 초기화
            case 'he':
                self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)  # He 초기화
            case 'gaussian':
                self.weights = np.random.randn(n_inputs, n_neurons) * 0.01  # Gaussian 초기화
            case _:  # 파이썬에서 _는 아무것도 입력되지 않은 상태를 의미함
                self.weights = np.random.randn(n_inputs, n_neurons)  # 기본 랜덤 초기화

        self.biases = np.random.uniform(0, 1, (1, n_neurons))


    def forward(self, inputs):
        return np.dot(inputs, np.array(self.weights)) + self.biases
        '''     #match-case로 activation 고르기
        match activation_Func:  # 활성화 함수 선택
            case 'sigmoid':     # 확률이 필요할때 시그모이드 주로 사용
                return 1 / (1 + np.exp(-layers_outputs))
            case 'relu':        # 일반적으로 렐루 사용
                return np.maximum(0, layers_outputs)
            case 'tanh':
                return np.tanh(layers_outputs)
        '''

# 추상 클래스 정의 (활성화 함수들의 공통 구조)
class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass

# 활성화 함수 클래스들 (추상 클래스 상속)
class ReLUActivation(ActivationFunction):
    def forward(self, inputs):
        return np.maximum(0, inputs)

class SigmoidActivation(ActivationFunction):
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

class TanhActivation(ActivationFunction):
    def forward(self, inputs):
        return np.tanh(inputs)

# 활성화 함수 인스턴스화 (추상 클래스 기반)
relu_activation = ReLUActivation()
sigmoid_activation = SigmoidActivation()
tanh_activation = TanhActivation()


'''
# 샘플 데이터 생성
inputs, y = spiral_data(samples=2, classes=3)
plt.scatter(inputs[:, 0], inputs[:, 1], c=y, cmap='brg')
plt.show()
'''

inputs = np.linspace(0, 2 * np.pi, 100).reshape(-1,1)
y = np.sin(inputs)

# Dense 레이어 생성
Dense1 = Layer_Dense(1, 8, initialize_method='xavier')  # 위 샘플은 2차원공간에서 정의되기 때문에 인풋을2로 설정해야함
Dense2 = Layer_Dense(8, 8, initialize_method='xavier')  # 인자에 위 가중치 초기화 방법 입력추가가능
Dense3 = Layer_Dense(8, 1, initialize_method='xavier')  # 인자에 위 가중치 초기화 방법 입력추가가능

Dense1.weights = np.array([[0], [2.0], [4.0], [3.0], [2.0], [3.0], [4.0], [1.0]]).T
Dense1.biases  = np.array([[1, 1, 0.5, 0, 0, 0, 0, 0]])

Dense2.weights = np.random.randn(8,8)*2
Dense2.biases  = np.zeros((1,8))

Dense3.weights = np.random.randn(8,1)*2
Dense3.biases  = np.zeros((1,1))

# 순전파
output1 = Dense1.forward(inputs)
Act_output1 = sigmoid_activation.forward(output1)

output2 = Dense2.forward(Act_output1)
Act_output2 = tanh_activation.forward(output2)

output3 = Dense3.forward(Act_output2)
Act_output3 = tanh_activation.forward(output3)

# print(Act_output3)

plt.plot(inputs, y, label="True Sine Wave", color="blue")
plt.plot(inputs, Act_output3, label="DNN Output", color="red")
plt.legend()    # 축의 각 색깔이 무엇을 의미하는지
plt.title("Sine Wave Approximation using Neural Network")
plt.show()


