#!/usr/bin/env python

from mlp import MLP
import random


class Trainer(object):

    def __init__(self, num_hidden, training_file, training_ratio=1.0):
        self.num_hidden = num_hidden
        self.testcases = self.parse_input(training_file)

        # a ratio of 1.0 means we train on 100% of the data, leaving no
        # "unseen" test cases. This could cause overfitting to data though.
        self.testcase_ratio = training_ratio

        num_inputs = len(self.testcases[0][0])
        num_outputs = len(self.testcases[0][1])

        self.network = MLP(num_inputs, num_hidden, num_outputs)

    def convert_to_network(self, line):
        """
        Converts data from input file format to
        the binary format favored by the network.
        """
        raise NotImplementedError()

    def convert_from_network(self, data):
        """
        Converts the output of the network to
        the format of the data being worked with.
        """
        raise NotImplementedError()

    def parse_input(self, file_path):
        """
        Parses the given file into a format suitable
        for use with the neural network.
        """
        with open(file_path, "r") as f:
            lines = f.read().strip().replace("\r", "").split("\n")

        random.shuffle(lines)

        return map(self.convert_to_network, lines)

    def train(self, learning_rate, max_timesteps=None, error_threshold=None):
        """
        Trains the neural network until a given stopping condition
        is met. Possible stopping conditions are:
            * the number of timesteps (i.e. stop after x iterations)
            * the error rate (i.e. stop after the error rate is
                sufficiently low).
        """
        if not max_timesteps and not error_threshold:
            raise Exception("Need to specify a stopping condition!")

        timesteps = 0
        error_trend = []
        slice_index = int(len(self.testcases) * self.testcase_ratio)
        testcases_slice = self.testcases[:slice_index]

        while True:
            timestep_error = 0.0
            random.shuffle(testcases_slice)

            for (input_data, output_data) in testcases_slice:
                self.network.forward(input_data)
                timestep_error += self.network.backwards(output_data)
                self.network.update_weights(learning_rate)

            error_trend.append(timestep_error)

            print "Error at timestep %d was %f." % (timesteps, timestep_error,)

            if error_threshold and (timestep_error <= error_threshold):
                return error_trend

            timesteps += 1
            if max_timesteps and (timesteps >= max_timesteps):
                return error_trend
