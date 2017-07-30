#!/usr/bin/env python

import cPickle
from trainer import Trainer


class NumberTrainer(Trainer):
    """
    This uses the abstract `Trainer` class to train an MLP
    specificially on the dataset we want.

    We can optionally pickle and save the trained MLP for later use.
    """

    def __init__(self, num_hidden, file_path):
        # this tells the trainer to only train on a random
        # 80% of the input data, so we can use the remaining
        # 20% to get an idea of the networks performance
        # against completely unseen data.
        test_ratio = 0.8
        super(NumberTrainer, self).__init__(num_hidden, file_path, test_ratio)

    def convert_to_network(self, line):
        """
        Each line of data contains 256 floats followed by 10 ints.
        E.g. 1.00 0.00 1.00 0.00 ... 0 0 0 0 0 1 0

        The first 256 floats represent the pixels of the image.
        1 being a black pixel, 0 being white.

        The 10 ints are flags with the correct output being 1.
        e.g. 0 0 0 1 0 0 0 -> 3, because the 3rd (zero indexed)
        value is 1.
        """
        line = map(lambda x: int(float(x)), line.strip().split(" "))
        return (line[:256], line[256:],)

    def convert_from_network(self, data):
        """
        Ideally, the output of the network should be nine zeros,
        with a single one, but alas, we get ten floats instead.

        These kind of represent the "best guess" of the network,
        so will often have many numbers < 0.01, and one > 0.5.

        We take the largest overall number as the final answer
        of the network.
        """
        return data.index(max(data))

    def error_against_unseen_data(self):
        """
        The `testcase_ratio` defines how the input data is split between
        the training data, and the testing data.

        Here we count how many of the unseen testcases are correctly
        determined to get an idea of the performance of the neural network.
        """
        if self.testcase_ratio == 1.0:
            raise Exception("Cannot calculate the error against unseen "
                            "data when the test ratio == 1.0.")

        slice_index = int(len(self.testcases) * self.testcase_ratio)
        testcases_slice = self.testcases[slice_index:]
        incorrect_count = 0

        for (input_data, output_data) in testcases_slice:
            self.network.forward(input_data)

            predicted = self.convert_from_network(self.network.output_layer)
            actual = self.convert_from_network(output_data)

            if predicted != actual:
                incorrect_count += 1

        return 100.0 * incorrect_count / len(testcases_slice)


if __name__ == '__main__':

    number_trainer = NumberTrainer(50, "letter_data.in")
    learning_rate = 0.1

    untrained_accuracy = 100.0 - number_trainer.error_against_unseen_data()
    print "Before training, the MLP has an accuracy of %.2f%%" % (untrained_accuracy)

    number_trainer.train(learning_rate, max_timesteps=20)
    # number_trainer.train(learning_rate, error_threshold=50)  # slow

    trained_accuracy = 100.0 - number_trainer.error_against_unseen_data()
    print "After training, the MLP has an accuracy of %.2f%%" % (trained_accuracy)

    # dump trained network for copying to webpage folder
    with open("trained_network.pickle", "w") as fout:
        cPickle.dump(number_trainer.network, fout)
