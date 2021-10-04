import numpy as np
import matplotlib.pyplot as plt


def hardlims(x):
    x[x >= 0] = 1
    x[x < 0] = -1
    return x


p1 = np.array([[-1, 1, 1, 1, -1],
               [1, -1, -1, -1, 1],
               [1, -1, -1, -1, 1],
               [1, -1, -1, -1, 1],
               [1, -1, -1, -1, 1],
               [-1, 1, 1, 1, -1]])
p2 = np.array([[-1, 1, 1, -1, -1],
               [-1, -1, 1, -1, -1],
               [-1, -1, 1, -1, -1],
               [-1, -1, 1, -1, -1],
               [-1, -1, 1, -1, -1],
               [-1, -1, 1, -1, -1]])
p3 = np.array([[1, 1, 1, -1, -1],
               [-1, -1, -1, 1, -1],
               [-1, -1, -1, 1, -1],
               [-1, 1, 1, -1, -1],
               [-1, 1, -1, -1, -1],
               [-1, 1, 1, 1, 1]])
p1_ravel, p2_ravel, p3_ravel = p1.reshape([30, 1]), p2.reshape([30, 1]), p3.reshape([30, 1])
w = p1_ravel @ p1_ravel.T + p2_ravel @ p2_ravel.T + p3_ravel @ p3_ravel.T

p1_half = np.array([[-1, 1, 1, 1, -1],
                    [1, -1, -1, -1, 1],
                    [1, -1, -1, -1, 1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1]])
p1_output = hardlims(w @ p1_half.reshape([30, 1]))
print(p1_output.reshape([6, 5]))

figsize = (12, 6)
fontsize = 30
plt.figure(figsize=figsize)
plt.rcParams.update({'font.size': fontsize})
plt.subplot(1, 2, 1)
plt.imshow(p1_half)
plt.title('Input')
plt.subplot(1, 2, 2)
plt.imshow(p1_output.reshape([6, 5]))
plt.title('Output')
plt.axis('off')
plt.tight_layout()
plt.savefig('img_name.png', dpi=300, bbox_inches='tight')

plt.show()
