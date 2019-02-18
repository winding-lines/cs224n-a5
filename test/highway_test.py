import unittest
import torch
import torch.optim as optim
import numpy as np

from highway import Highway

class HighwayTest(unittest.TestCase):
    def setUp(self):
        print("------\ntest setup()\n-----\n")
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed * 13 // 7)

        # setup the device
        self.device = torch.device("cpu")

    def initializeWeights(hwy: Highway, projection:float, gate:float):
        """Initialize all the weights in the projection and gate level to the same value.
        """

        # initialize the project layer to halve the input data
        hwy.projLinear.weight.data.fill_(projection)
        hwy.projLinear.bias.data.fill_(0.0)
        hwy.gateLinear.weight.data.fill_(gate)
        hwy.gateLinear.bias.data.fill_(0.0)

    def testProjection(self):
        hwy = Highway(1,1)
        input = torch.tensor([1.0])

        HighwayTest.initializeWeights(hwy, 0.3,10) 

        output = hwy.forward(input)
        torch.testing.assert_allclose(output, [0.3])

        HighwayTest.initializeWeights(hwy, 20,10) 

        output = hwy.forward(input)
        torch.testing.assert_allclose(output, [20])

    def testGating(self):
        hwy = Highway(1,1)
        input = torch.tensor([1.0])

        HighwayTest.initializeWeights(hwy, 20,-10) 

        output = hwy.forward(input)
        torch.testing.assert_allclose(output, [1.0], 1e-3, 1e-3)


