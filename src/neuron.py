# -*- coding: utf-8 -*-

from random import choice
from numpy import dot, random


class Neuron:

    def __init__(self, eta, input_size, unit_step):
        self.eta = eta
        self.weights = random.rand(input_size)
        self.unit_step = unit_step

    def train(self, training_data, times):
        for i in xrange(times):
            inputs, expected = choice(training_data)
            result = dot(self.weights, inputs)
            error = expected - self.unit_step(result)
            self.weights += self.eta * error * inputs

    def get_result(self, inputs):
        result = dot(inputs, self.weights)
        return result

    def get_unit_step(self, inputs):
        result = self.get_result(inputs)
        return self.unit_step(result)
