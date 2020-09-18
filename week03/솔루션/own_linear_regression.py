import numpy as np

x_train = [1., 2., 3., 4.]
y_train = [0., -1., -2., -3.]

def loss(w, x_list, y_list):
    N = len(x_list)
    val = 0.0
    for i in range(N):
        val += (w[0] * x_list[i] + w[1] - y_list[i])**2 / N
    return val

def grad_loss(w, x_list, y_list):
    dim = len(w)
    N = len(x_list)
    val = np.array([0.0, 0.0])
    for i in range(N):
        er = w[0] * x_list[i] + w[1] - y_list[i]
        val += 2.0 * er * np.array([x_list[i], 1.0]) / N
    return val

MaxIter = 10
learning_rate = 0.01
w0 = np.array([.3, -.3])
for i in range(MaxIter):
	grad = grad_loss(w0, x_train, y_train)
	w1 = w0 - learning_rate * grad
	print(i, w0, loss(w0, x_train, y_train))
	w0 = w1

print(i+1, w0, loss(w0, x_train, y_train))