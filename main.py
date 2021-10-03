from typing import List

import numpy as np

from layer import Dense


class NeuralModel:
    layers: List[Dense] = []

    def __init__(self):
        pass

    def add(self, layer):
        if layer.output_shape is not None:
            layer.init_parameter()
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-2].output_shape = self.layers[-1].input_shape
            self.layers[-2].init_parameter()

    def predict(self, x):
        result_predict = self.layers[0](x)
        for layer in self.layers[1:]:
            result_predict = layer(result_predict)
        return result_predict

    def summary(self):
        for layer in self.layers:
            print(layer.input_shape, layer.output_shape, layer.bias.shape, layer.weight.shape)

    def train(self, x, y):
        epoch = 0

        layer_prediction = self.layers[0](x.T)

        print(f'x = {x}')
        print(f'y = {y}')
        while not np.array_equal(self.predict(x.T), y):
            print(f'predict: {self.predict(x.T)}')
            print(f'y: {y}')
            epoch += 1
            print('epoch = ', epoch)
            for idx_layer, layer in enumerate(self.layers):
                if idx_layer != 0:
                    layer_prediction = layer(layer_prediction.T)
                for idx_x, x_val in enumerate(x):
                    error = (y[0][idx_x] - self.predict(x.T)[0][idx_x])
                    # print(f'predict(x.T) = {self.predict(x.T)}')
                    print(f'y[0][idx_x] = {y[0][idx_x]}')
                    print(f'self.predict(x.T)[0][idx_x] = {self.predict(x.T)[0][idx_x]}')
                    print(f'error: {error}')
                    print(f'weight before = {layer.weight}')
                    print(f'bias berfore = {layer.bias}')
                    layer.weight += np.multiply(error, x_val).T
                    layer.bias += error
                    print(f'weight after = {layer.weight}')
                    print(f'bias after = {layer.bias}')
                    print()


neural_model = NeuralModel()
neural_model.add(Dense(2, 1, activation='Hardlim'))
# dataset = np.array([[1, -1, -1], [1, 1, -1]])
# label = np.array([[0, 1]])

dataset = np.array([[2, 2], [1, -2], [-2, 2], [-1, 1]])
label = np.array([[0, 1, 0, 1]])

neural_model.train(dataset, label)
result = neural_model.predict(dataset.T)
print(result)
# df = pd.read_csv('train.csv')

# g = np.array([[1.35, 0.3]])
# p = -g
# A = np.array([[2,1],[1,2]])
# d = p@A@p.T
# u = p@p.T
# print(u)
