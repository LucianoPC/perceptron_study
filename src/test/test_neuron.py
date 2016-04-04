#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np

from src.neuron import Neuron


class NeuronTests(unittest.TestCase):

    def test_one_neuron(self):
        unit_step = (lambda inputs, weights:
                     0 if np.dot(inputs, weights) < 0 else 1)

        training_data = [
            (np.array([0, 0, 1]), 0),
            (np.array([0, 1, 1]), 1),
            (np.array([1, 0, 1]), 0),
            (np.array([1, 1, 1]), 1),
        ]

        eta = 0.2
        input_size = 3
        training_times = 1000000

        neuron = Neuron(eta, input_size, unit_step)
        neuron.train(training_data, training_times)

        results = []
        assertions = [element[1] for element in training_data]
        for x, _ in training_data:
            unit = neuron.get_unit_step(x)
            results.append(unit)

        self.assertEqual(assertions, results)
