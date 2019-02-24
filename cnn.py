#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, in_channel: int, out_channels: int, kernel_size:int = 5):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channel, out_channels=out_channels, kernel_size=kernel_size)

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv1d(input)

    def initializeUniform(self, value: float):
        with torch.no_grad():
            self.conv1d.weight.data.fill_(value)
            self.conv1d.bias.data.fill_(0.0)

### END YOUR CODE

