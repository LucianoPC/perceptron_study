#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import random
from random import choice


class Neuron:

    def __init__(self, eta, input_size, unit_step):
        self.eta = eta
        self.weights = random.rand(input_size)
        self.unit_step = unit_step

    def train(self, training_data, times):
        for i in xrange(times):
            inputs, self.expected = choice(training_data)
            self.error = self.expected - self.unit_step(inputs, self.weights)
            self.weights += self.eta * self.error * inputs

    def get_unit_step(self, inputs):
        return self.unit_step(inputs, self.weights)
