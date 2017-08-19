import numpy as np
import scipy.special as sci


class NeuralNetwork:
    def __init__(self, num_inodes, num_hnodes, num_onodes, learning_rate):
        bg -0.5),
            (self.hnodes, self.inodes)
        )
        self.who_vals = np.random.normal(
            0.0,
            pow(self.hnodes, -0.5),
            (self.onodes, self.hnodes)
        )

        self.activation_function = lambda x: sci.expit(x)
        self.inverse_activation_function = lambda x: sci.logit(x)

    def _calculate_layer_output(self, inputs, layer_matrix):
        layer_vals = np.dot(layer_matrix, inputs)
        return self.activation_function(layer_vals)

    def _think(self, inputs):
        hidden_result = self._calculate_layer_output(inputs, self.wih_vals)
        output_result = self._calculate_layer_output(hidden_result.T, self.who_vals)
        return output_result

    def thought_process(self, inputs):
        output_reverse = self.inverse_activation_function(inputs)
        output_reverse_result = np.dot(output_reverse, self.who_vals)

        output_reverse_result -= np.min(output_reverse_result)
        output_reverse_result /= np.max(output_reverse_result)
        output_reverse_result *= 0.98
        output_reverse_result += 0.01

        hidden_reverse = self.inverse_activation_function(output_reverse_result)
        hidden_reverse_result = np.dot(hidden_reverse, self.wih_vals)

        hidden_reverse_result -= np.min(hidden_reverse_result)
        hidden_reverse_result /= np.max(hidden_reverse_result)
        hidden_reverse_result *= 0.98
        hidden_reverse_result += 0.01

        return hidden_reverse_result

    def train(self, inputs, targets):
        hidden_result = self._calculate_layer_output(inputs, self.wih_vals)
        output_result = self._calculate_layer_output(hidden_result.T, self.who_vals)

        output_error = targets - output_result
        hidden_error = np.dot(self.who_vals.T, output_error)

        self.who_vals += self.lr * np.outer(
            (output_error * output_result * (1 - output_result)),
            hidden_result
        )
        self.wih_vals += self.lr * np.outer(
            (hidden_error * hidden_result * (1 - hidden_result)),
            inputs
        )

    def query(self, inputs):
        return self._think(inputs)
