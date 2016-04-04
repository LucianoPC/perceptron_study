#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import array, dot
from neuron import Neuron

unit_step = lambda inputs, weights: 0 if dot(inputs, weights) < 0 else 1

training_data = [
    (array([0, 0, 1]), 0),
    (array([0, 1, 1]), 1),
    (array([1, 0, 1]), 0),
    (array([1, 1, 1]), 1),
]

eta = 0.2
input_size = 3
training_times = 100

neuron = Neuron(eta, input_size, unit_step)
neuron.train(training_data, training_times)

for x, _ in training_data:
    unit = neuron.get_unit_step(x)
    print("{}: -> {}".format(x, unit))
