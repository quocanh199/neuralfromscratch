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
        if len(self.layers) > 1:
            for layer in self.layers[1:]:
                result_predict = layer(result_predict)
        return result_predict

    def train(self, x, y, learning_rule):
        epoch = 0
        layer_prediction = self.layers[0](x)

        print(f'x = {x}')
        print(f'y = {y}')
        while not np.array_equal(self.predict(x), y):
            prediction = self.predict(x)
            print(f'predict: {prediction}')
            print(f'y: {y}')
            epoch += 1
            print('epoch = ', epoch)
            for idx_layer, layer in enumerate(self.layers):
                if idx_layer != 0:
                    layer_prediction = layer(layer_prediction)
                print(f'weight before = {layer.weight}')
                if layer.output_shape:
                    if learning_rule == 'basic_rule':
                        layer.basic_rule(t=y, p=x)
                    elif learning_rule == 'learning_rate':
                        layer.learning_rate(t=y, p=x, alpha=1)
                    elif learning_rule == 'filtered_learning_rate':
                        layer.filtered_learning_rate(t=y, p=x, alpha=1, gamma=0)
                    elif learning_rule == 'delta_rule':
                        layer.delta_rule(t=y, p=x, a=prediction, alpha=1)
                    elif learning_rule == 'unsupervised_hebb':
                        layer.unsupervised_hebb(a=prediction, p=x, alpha=1)
                print(f'weight after = {layer.weight}')
                print()

    def summary(self):
        for layer in self.layers:
            print(layer.input_shape, layer.output_shape, layer.bias.shape, layer.weight.shape)


neural_model = NeuralModel()
dataset = np.array([[0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5]]).T
label = np.array([[1, -1], [1, 1]]).T

neural_model.add(Dense(4, 2, activation='Hardlim'))
neural_model.train(x=dataset, y=label, learning_rule='unsupervised_hebb')
result = neural_model.predict(dataset)

print(f'result = {result}')
