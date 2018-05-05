import numpy as np
import math
import pickle


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Function(object):
    def calc(self, x):
        raise NotImplemented

    def dcalc(self, x, y):
        raise NotImplemented


class Sigmoid(Function):
    def calc(self, x):
        return (lambda x: 1 / (1 + np.exp(-x)))(x)

    def dcalc(self, x, y):
        return self.calc(x)*(1 - self.calc(x))


class TwoLayeredNN(object):
    def __init__(self, layers_func, learning_rate, num_classes, num_epocs):
        self.eta = learning_rate
        self.epocs = num_epocs
        self.num_classes = num_classes
        self.function_layers = layers_func["layers_func"]
        self.num_of_layers = len(self.function_layers)
        self.layer_weights_and_b_pairs = []
        for x_size, y_size in layers_func["sizes"]:
            self.layer_weights_and_b_pairs.append((np.random.rand(x_size, y_size), np.random.rand(1, y_size)))

    def __forward_in_net__(self, train_example):
        input , tag = train_example
        zs = []
        hs = []
        input_temp = input
        for function_activation, weight_and_b in zip(self.function_layers, self.layer_weights_and_b_pairs):
            W, b = weight_and_b
            z_i_temp = np.dot(input_temp, W) + b
            h_i_temp = function_activation.calc(z_i_temp)
            zs.append(z_i_temp)
            hs.append(h_i_temp)
        return zs, hs, hs[-1]

    def __backprop_in_net__(self, forward_net_info):
        zs, hs, model_output = forward_net_info
        return 0, 0

    def learn(self, training_set, batch_size=1):
        for _ in self.epocs:
            for batch in list(chunks(training_set, batch_size)):
                sum_weight_change, sum_bias_change = (0, 0)
                for train_example in batch:
                    zs, hs, model_output = self.__forward_in_net__(train_example)
                    dW, dB = self.__backprop_in_net__((zs, hs, model_output))
                    sum_weight_change += dW
                    sum_bias_change += dB

    def compare_against_test_set(self, test_set):
        num_right = 0
        for test in test_set:
            input, tag = test
            _, _, model_output = self.__forward_in_net__(input)
            prediction = np.argmax(model_output) + 1
            if prediction == tag:
                num_right += 1
        return num_right / len(test_set)


if __name__ == '__main__':
    train_x, train_y, test_x = pickle.loads(open("./pickled.bin", "rb").read())
    function_sigmoid = Sigmoid()
    proper_data_set = zip(train_x, train_y)
    np.random.shuffle(proper_data_set)
    real_train_80, validation_20 = proper_data_set[:int(0.8 * len(train_x))], proper_data_set[int(0.8 * len(train_x)):]
    network = TwoLayeredNN(learning_rate=0.1,
                           num_classes=10,
                           layers_func={"layers_func": [function_sigmoid, function_sigmoid],
                                        "sizes": [(784, 16), (16, 10)]})
