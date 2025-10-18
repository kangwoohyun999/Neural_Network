import sys, os, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist

# 활성화 함수 및 소프트맥스 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# MNIST 데이터 불러오기
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 네트워크 가중치 불러오기
def init_network():
    with open(os.path.dirname(__file__) + "/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

# 예측 함수
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

# 데이터와 네트워크 초기화
x, t = get_data()
network = init_network()

# 1. 숫자 8의 정확도 계산
target_digit = 8
correct_cnt_8 = 0
total_cnt_8 = 0

for i in range(len(x)):
    if t[i] == target_digit:
        total_cnt_8 += 1
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == target_digit:
            correct_cnt_8 += 1

accuracy_8 = correct_cnt_8 / total_cnt_8
print(f"Digit {target_digit} Accuracy: {accuracy_8:.4f}")

# 2. 숫자 0~9 각각의 정확도 계산
digit_accuracies = {}

for digit in range(10):
    correct_cnt = 0
    total_cnt = 0
    for i in range(len(x)):
        if t[i] == digit:
            total_cnt += 1
            y = predict(network, x[i])
            p = np.argmax(y)
            if p == digit:
                correct_cnt += 1
    accuracy = correct_cnt / total_cnt
    digit_accuracies[digit] = accuracy
    print(f"Digit {digit} Accuracy: {accuracy:.4f}")

# 가장 정확도가 낮은 숫자 출력
lowest_digit = min(digit_accuracies, key=digit_accuracies.get)
print(f"\nLowest accuracy: Digit {lowest_digit} ({digit_accuracies[lowest_digit]:.4f})")
