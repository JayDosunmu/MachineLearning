#! /usr/bin/env python
import numpy as np
from neuralnetwork import NeuralNetwork as NN

expected_nums = {
    0: [0.99, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    1: [0.1, 0.99, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    2: [0.1, 0.1, 0.99, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    3: [0.1, 0.1, 0.1, 0.99, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    4: [0.1, 0.1, 0.1, 0.1, 0.99, 0.1, 0.1, 0.1, 0.1, 0.1],
    5: [0.1, 0.1, 0.1, 0.1, 0.1, 0.99, 0.1, 0.1, 0.1, 0.1],
    6: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.99, 0.1, 0.1, 0.1],
    7: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.99, 0.1, 0.1],
    8: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.99, 0.1],
    9: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.99]
}


def main():
    with open('mnist_test.csv', 'r') as f:
        training_file = f.readlines()

    training_data = []
    for line in training_file:
        line = line.split(',')
        training_data.append([int(line[0]), [int(num) for num in line[1:]]])

    with open('mnist_test_10.csv', 'r') as f:
        testing_file = f.readlines()

    testing_data = []
    for line in testing_file:
        line = line.split(',')
        testing_data.append([int(line[0]), [int(num) for num in line[1:]]])

    num_net = NN(784, 100, 10, 0.2)

    for data in training_data:
        num_net.train(data[1], expected_nums[data[0]])

    for i in range(10):
        test_input = testing_data[i]
        nn_output = num_net.query(test_input[1])
        guess = np.argmax(nn_output)
        print('Output values: ' + str(nn_output))
        print('Guess: ' + str(guess))
        print('Expected answer: ' + str(test_input[0]))
        error = expected_nums[i] - nn_output
        print(error)


if __name__ == '__main__':
    main()
