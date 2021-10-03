import numpy as np

from activation import ReLU, Sigmoid, Orgin, Hardlim


class Dense:
    weight = None
    bias = None
    input_shape = None
    output_shape = None
    activation = None

    def __init__(self, input_shape, output_shape=None, bias=True, activation=activation):

        self.input_shape = input_shape
        self.output_shape = output_shape
        if bias is False:
            self.bias = 0
        if activation == 'ReLU':
            self.activation = ReLU()
        elif activation == 'Sigmoid':
            self.activation = Sigmoid()
        elif activation == 'Origin':
            self.activation = Orgin()
        elif activation == 'Hardlim':
            self.activation = Hardlim()

    def __call__(self, x):
        return self.activation(np.dot(self.weight, x) + self.bias)

    def init_parameter(self):
        # self.weight = np.random.rand(self.output_shape, self.input_shape)
        # self.weight = np.array([[0.5, -1, -0.5]])
        self.weight = np.array([[0,0]])
        if self.bias is None:
            # self.bias = np.random.rand(self.output_shape, 1)
            # self.bias = np.array([[0.5]])
            self.bias = np.array([[1]])

