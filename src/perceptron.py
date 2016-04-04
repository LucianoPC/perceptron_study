#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import array, dot
from neuron import Neuron

unit_step_a = lambda inputs, weights: 0 if dot(inputs, weights) < 0 else 1
unit_step_b = lambda inputs, weights: 1 if dot(inputs, weights) < 0 else 2
unit_step_c = lambda inputs, weights: 0 if dot(inputs, weights) < 0 else 2

unit_step_d = lambda inputs, weights: dot(inputs, weights)

training_data = [
    (array([0, 0, 1]), 0),
    (array([0, 1, 1]), 1),
    (array([1, 0, 1]), 0),
    (array([1, 1, 1]), 1),
]

eta = 0.2
input_size = 3
training_times = 100

neuron_a = Neuron(eta, input_size, unit_step_a)
neuron_a.train(training_data, training_times)

neuron_b = Neuron(eta, input_size, unit_step_b)
neuron_b.train(training_data, training_times)

neuron_c = Neuron(eta, input_size, unit_step_c)
neuron_c.train(training_data, training_times)

neurons = [neuron_a, neuron_b, neuron_c]

new_training_data = []

for x, label in training_data:
    print("=")
    unities = []
    for neuron in neurons:
        result = dot(x, neuron.weights)
        unit = neuron.get_unit_step(x)
        unities.append(unit)
        print("{}: {} -> {}".format(x, result, unit))

    new_training_data.append((array(unities), label))

neuron_d = Neuron(eta, input_size, unit_step_d)
neuron_d.train(new_training_data, training_times)

for x, label in new_training_data:
    print("\n#####")
    unit = neuron_d.get_unit_step(x)
    unit = int(unit)
    print("expected: {}, error: {}".format(neuron_d.expected, neuron_d.error))
    print("{}: -> {}".format(x, unit))
