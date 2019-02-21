#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import torch

from highway import Highway

class HighwayTest(unittest.TestCase):
    def setUp(self):
        # setup the device
        self.device = torch.device("cpu")

    def testProjection(self):
        hwy = Highway(1,1)
        input = torch.tensor([1.0])

        hwy.initializeWeights(0.3,10) 

        output = hwy.forward(input)
        torch.testing.assert_allclose(output, [0.3])

    def testProjection2(self):
        hwy = Highway(1,1)
        input = torch.tensor([1.0])
        hwy.initializeWeights(20,10) 

        output = hwy.forward(input)
        torch.testing.assert_allclose(output, [20])

    def testGating(self):
        hwy = Highway(1,1)
        input = torch.tensor([1.0])

        hwy.initializeWeights(20,-10) 

        output = hwy.forward(input)
        torch.testing.assert_allclose(output, [1.0], 1e-3, 1e-3)


