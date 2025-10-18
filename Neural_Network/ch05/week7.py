import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.mnist import load_mnist
from ch0102.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_train = x_train[:, ::4]
x_test = x_test[:, ::4]

network = TwoLayerNet(input_size=196, hidden_size=2, output_size=10)

iters_num = 3000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % 20 == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"반복 {i:4d}: train acc = {train_acc:.4f}, test acc = {test_acc:.4f}, loss = {loss:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(train_loss_list)), train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("손실 함수 추이")

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(train_acc_list)) * 20, train_acc_list, label='train acc')
plt.plot(np.arange(len(test_acc_list)) * 20, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("iteration")
plt.ylabel("accuracy")
plt.title("정확도 추이")
plt.legend()
plt.tight_layout()
plt.show()
