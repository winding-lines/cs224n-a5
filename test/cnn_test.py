#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import torch

from cnn import CNN

class CnnTest(unittest.TestCase):
    def setUp(self):
        # setup the device
        self.device = torch.device("cpu")

    def testForward(self):
        cnn = CNN(1,1)
        input = torch.tensor([[[1.0,2.0,3.0,4.0,5.0,6.0]]])
        cnn.initializeUniform(1.0)
        output = cnn(input)
        print(f"output {output}")
        torch.testing.assert_allclose(output, [[[15.3284, 20.3284]]])