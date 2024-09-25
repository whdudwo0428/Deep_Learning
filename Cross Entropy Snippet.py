import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt     # 그래프 보려고 사용

nnfs.init() # 랜덤시드들이 고정됨 /이 기준으로만 랜덤값이 설정됨

### 1. Define the Dense Layer!!!
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialize_method='random'):  # 입력갯수 출력갯수 / init의 역할은 생성자! 최초로 불러질 때 한 번 실행되는 내용
        '''   걍 함수 밑에 주석땡땡땡 엔터 하면 자동으로 생김!! 개신기함
        :param n_inputs: 입력의 개수
        :param n_neurons: Layer 내의 뉴련의 개수
        :param initialize_method:
        '''
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
        #type1 : return을 사용하기 싫은경우
        #self.output = np.dot(inputs, np.array(self.weights)) + self.biases
        # type2 : return을 사용하고 싶은경우
        # retrun np.dot(inputs, np.array(self.weights)) + self.biases

### 2. Activation Class
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

### 3. Activation Softmax Class
#Softmax :주어진 입력값을 0과 1 사이의 값으로 변환하여 각 클래스에 속할 확률을 계산
class Softmax:
    def forward(self, predictions, targets):
        #####  predictions, targets 두개를 정의해서 넣는거임
        softmax_outputs = np.array([
            [0.7, 0.1, 0.2],
            [0.1, 0.5, 0.4],
            [0.2, 0.2, 0.6]
        ])
        targets = np.array([0, 1, 2])
                                                #softmax_outputs 변수명 써보기
        loss = forward(predictions, targets)
        print("Categorical Cross-Entropy Loss:", loss)

### 4. Loss_Categorical_Cross_entropy
class Loss_categorical_cross_entropy:
    def forward(self, predictions, targets):
        '''
        :param predictions: Dense Later output -> Softmax 취한 출력
        :param targets: 정답지, one-hot encoding
        :return: categorical cross entropy loss 연산값
        '''

        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions,1e-7, 1 - 1e-7)
        # clip : 자르다! 식을 보면 로그가 있는데 로그0은 없잖, 정의x
        # 컴터가 알아먹을 수 있게 "범위를 정해줌"
        ''' e는 10을 의미 e-7 = 10^7
        if predictions == 0:
            predictions = 1e-7
        '''

        # If targets are sparse
        if targets.ndim == 1:
            correct_confidences = predictions[np.arange(len(predictions)), targets]
        else:
            # if targets are one-hot encoded
            correct_confidences = np.sum(predictions * targets, axis=1)

        # Calculate negative log likehood
        negative_log_likehood = - np.log(correct_confidences)

        # Calulate avergae loss
        return np.mean(negative_log_likehood)

### 5. Create dataset
inputs, y = spiral_data(samples=100, classes=3)
#plt.scatter(inputs[:, 0], inputs[:, 1], c=y, cmap='brg')
#plt.show()

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
output1 = sigmoid_activation.forward(Dense1.forward(inputs))
output2 = relu_activation.forward(Dense2.forward(output1))
output3 = tanh_activation.forward(Dense3.forward(output2))

### 6. forward

### 7. loss calculation

### 8. print loss

# Dense 레이어 생성


plt.plot(inputs, y, label="True Sine Wave", color="blue")
plt.plot(inputs, output3, label="DNN Output", color="red")
plt.legend()    # 축의 각 색깔이 무엇을 의미하는지
plt.title("Sine Wave Approximation using Neural Network")
plt.show()


