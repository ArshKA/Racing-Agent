import numpy as np
import math

class NPModel:
    def __init__(self, weights):
        self.weights = weights
        self.reward = 0

    def __repr__(self):
        string = ""
        for layer in self.weights:
            string += "Weights: " + str(layer[0].tolist()) + "\nBias: " + str(layer[1]) + "\n" + '-' * 150 + "\n"
        return string

    def forward(self, output):
        for layer in self.weights:
            output = np.matmul(layer[:-1], output)[0] + layer[-1]
        
        # CHANGE BASED ON ENV ACTION SPACE
        return [np.tanh(output[0]), self.sigmoid(output[1]), self.sigmoid(output[2])]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def sigmoid(self, x):
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        else:
            return 1 / (1 + math.exp(-x))

    def generate_offspring(self, n=1, scale=1):
        all_children = []
        for i in range(n):
            child_weights = []
            for layer in self.weights:
                child_weights.append(
                    (layer[0] + np.random.normal(scale=scale, size=layer[0].shape), layer[1] + np.random.normal(scale=scale)))
            all_children.append(NPModel(child_weights))
        return all_children
