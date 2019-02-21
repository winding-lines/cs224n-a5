#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Highway(nn.Module):
    """Highway Networks, Srivastava et al., 2015 https://arxiv.org/abs/1505.00387
    """

    def __init__(self, in_features: int, out_features: int, has_relu:bool=True):
        """
        Create the primitives used to build the Highway network.

        @param in_features: size of each input sample 
        @param out_features: size of the output sample
        """
        super(Highway, self).__init__()
        self.projLinear = nn.Linear(in_features=in_features, out_features=out_features)
        self.gateLinear = nn.Linear(in_features=in_features, out_features=out_features)
        self.has_relu = has_relu
         

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the input through the highway.
        """
        x_proj = self.projLinear(input)
        x_proj = F.relu(x_proj) if self.has_relu else x_proj
        x_gate = torch.sigmoid(self.gateLinear(input))
        return x_gate * x_proj + (1-x_gate) * input

    def initializeWeights(self, projection:Optional[float], gate:float):
        """Initialize all the weights in the projection and gate level to the same value.
        """
        with torch.no_grad():
            # initialize the projection layer
            if projection is None:
                torch.nn.init.xavier_uniform(self.projLinear.weight)
            else:
                self.projLinear.weight.data.fill_(projection)
                self.projLinear.bias.data.fill_(0.0)

            # initialize the gate layer
            self.gateLinear.weight.data.fill_(gate)
            self.gateLinear.bias.data.fill_(0.0)



### END YOUR CODE 

