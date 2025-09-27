import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y1 = exp_a/sum_exp_a
    
    return y1

def init_network():
    network = {}
    network['W1'] = np.array([[-1,1],[1,-2],[2,1]])
    
    return network

def forward(network, x):
    W1 = network['W1']
    
    a1 = np.dot(x, W1)
    y = softmax(a1)
    
    return y

network = init_network()
x = np.array([1.0,9.0,4.0])
y = forward(network, x)

print(y)
