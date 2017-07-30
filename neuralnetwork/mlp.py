#!/usr/bin/env python

import math
import random


class MLP(object):
    """
    This is an implementation of a very basic Multi-Layered Perceptron (MLP).
    Activation function is a basic sigmoid: 1 / (1 + e^-x)

    More info on the inner workings of an MLP can be found in the README.md.
    """

    def __init__(self, num_inputs, num_hidden, num_outputs):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # holds the values of the input, output, and hidden nodes
        self.input_layer = [0 for i in range(num_inputs + 1)]
        self.hidden_layer = [0 for i in range(num_hidden + 1)]
        self.output_layer = [0 for i in range(num_outputs)]

        # holds the weight of each edge between any two nodes
        self.lower_weights = [[(random.random() - 0.5) for i in range(num_inputs + 1)] for j in range(num_hidden + 1)]
        self.upper_weights = [[(random.random() - 0.5) for i in range(num_outputs)] for j in range(num_hidden + 1)]

        self.lower_weights_delta = [[0 for i in range(num_inputs + 1)] for j in range(num_hidden + 1)]
        self.upper_weights_delta = [[0 for i in range(num_outputs)] for j in range(num_hidden + 1)]

        self.lower_activations = [0 for i in range(num_hidden + 1)]
        self.upper_activations = [0 for i in range(num_outputs)]

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        """
        Foward propagation algorithm.
        """
        self.input_layer = map(float, inputs) + [1.0]

        # add the bias
        # self.input_layer[self.num_inputs] = 1.0
        self.hidden_layer[self.num_hidden] = 1.0

        # recalculate hidden layer
        for i in range(self.num_hidden):  # skips the bias
            self.lower_activations[i] = 0.0

            for j in range(self.num_inputs):  # uses the bias
                self.lower_activations[i] += self.lower_weights[i][j] * self.input_layer[j]

            self.hidden_layer[i] = self.sigmoid(self.lower_activations[i])

        # reclaculate output layer
        for i in range(self.num_outputs):  # has no bias
            self.upper_activations[i] = 0.0

            for j in range(self.num_hidden):  # uses the bias
                self.upper_activations[i] += self.upper_weights[j][i] * self.hidden_layer[j]

            self.output_layer[i] = self.sigmoid(self.upper_activations[i])

    def backwards(self, targets):
        """
        Error propagation, or you know, learning.

        Returns the sum of squared error between the target and output layer.
        """
        total_error = 0.0

        # sum of squared errors
        for i in range(self.num_outputs):
            total_error += (targets[i] - self.output_layer[i]) ** 2

        output_error = [0 for i in range(self.num_outputs)]

        # recalculate the delta of the weights between
        # the output layer, and the hidden layer
        for i in range(self.num_outputs):
            output_error[i] = self.output_layer[i] * \
                             (1 - self.output_layer[i]) * \
                             (targets[i] - self.output_layer[i])

            for j in range(self.num_hidden):
                self.upper_weights_delta[j][i] = output_error[i] * self.hidden_layer[j]

        # recalculate the delta of the weights between
        # the hidden layer, and the input layer
        for i in range(self.num_hidden):
            upper_error = 0.0

            for j in range(self.num_outputs):
                upper_error += output_error[j] * self.upper_weights[i][j]
            upper_error *= self.hidden_layer[i] * (1 - self.hidden_layer[i])

            for j in range(self.num_inputs - 1):
                self.lower_weights_delta[i][j] = upper_error * self.input_layer[j]

        return total_error

    def update_weights(self, learning_rate):
        """
        Adds weight deltas to weights, and resets deltas to zero.
        """
        for i in range(self.num_outputs):
            for j in range(self.num_hidden):
                self.upper_weights[j][i] += learning_rate * self.upper_weights_delta[j][i] * self.hidden_layer[j]
                self.upper_weights_delta[j][i] = 0.0
        
        for i in range(self.num_inputs):
            for j in range(self.num_hidden - 1):
                self.lower_weights[j][i] += learning_rate * self.lower_weights_delta[j][i] * self.input_layer[i]
                self.lower_weights_delta[j][i] = 0;
            
