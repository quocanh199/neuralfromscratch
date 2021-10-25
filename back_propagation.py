import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu

p = np.arange(-2, 2.1, 0.2)
t = 1 + np.sin(np.pi / 4 * p)
# plt.plot(p, t, 'o')
# plt.show()

# Khởi tạo trọng số
w1 = np.array([[-0.27, -0.41]]).T
b1 = np.array([[-0.48, -0.13]]).reshape((2, 1))
w2 = np.array([[0.09, -0.17]])
b2 = np.array([[0.48]])

# Huấn luyện mạng với 50 epochs
"""
Với mỗi iteration
1. Tính forward propagation
2. Tính back propagation
3. Cập nhật trọng số
"""


def log_sig(x):
    return 1 / (1 + np.exp(-x))


alpha = 0.1

for epoch in range(1):
    # cập nhật network input
    for idx in range(21):
        # tính forward propagation
        a1 = log_sig(w1 * p[idx] + b1)
        a2 = w2 @ a1 + b2
        print(f'a1 = {a1},\na2 = {a2}')

        # tính back propagation
        s2 = -2 * (t[idx] - a2)
        s1 = np.diag(((1 - a1) * a1).ravel()) @ w2.T * s2

        # cập nhật trọng số
        w2 -= alpha * s2 @ a1.T
        b2 -= alpha * s2
        w1 -= alpha * s1 * p[idx]
        b1 -= alpha * s1

