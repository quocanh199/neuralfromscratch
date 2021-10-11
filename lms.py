import numpy as np


# p1 = np.array([[1, -1, -1]])
# p2 = np.array([[1, 1, -1]])
#
# t1 = -1
# t2 = 1

def predict(weight, input):
    return weight @ input


alpha = 0.2
p = np.array([[1.0, -1.0, -1.0], [1.0, 1.0, -1.0]]).T
t = np.array([[-1.0, 1.0]])

w = np.array([[0.0, 0.0, 0.0]])
epoch = 0
while not np.array_equal(predict(w, p), t):
    epoch += 1
    print(f'epoch = {epoch}')
    w += 2 * alpha * (t - predict(w, p)) @ p.T
    print(f'weight = {w}')
    print(f'predict = {predict(w, p)}')
