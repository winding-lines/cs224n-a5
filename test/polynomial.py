# Build a test data set from a polynomial
#
# Adapted from https://github.com/pytorch/examples/tree/master/regression
from itertools import count

import torch
import torch.nn.functional as F

class Polynomial:
    """Builds a dataset by evaluating a polynomial.
    """

    def __init__(self, degree: int = 4):
        self.poly_degree = degree
        self.W_target = torch.randn(degree, 1) * 5
        self.b_target = torch.randn(1) * 5


    def make_features(self, x):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        x = x.unsqueeze(1)
        return torch.cat([x ** i for i in range(1, self.poly_degree+1)], 1)


    def f(self, x):
        """Approximated function."""
        return x.mm(self.W_target) + self.b_target.item()

    def poly_desc(W, b):
        """Creates a string description of a polynomial."""
        result = 'y = '
        for i, w in enumerate(W):
            result += '{:+.2f} x^{} '.format(w, len(W) - i)
        result += '{:+.2f}'.format(b[0])
        return result

    def get_batch(self, batch_size=32):
        """Builds a batch i.e. (x, f(x)) pair."""
        random = torch.randn(batch_size)
        x = self.make_features(random)
        y = self.f(x)
        return x, y

    def degree(self):
        """Return the degree of the polynomial.
        """
        return self.W_target.size(0)

    def train(self, model:torch.nn.Module, lr:float=0.001) -> (int, float):
        """ Train model

        @param model (torch.nn.Module): model to train
        @param lr (float): learning rate

        Example:

            poly = Polynomial(4)
            fc = torch.nn.Linear(poly.degree(), 1)
            poly.linear_train(fc)
        """
        for batch_idx in range(1,10000):
            # Get data
            batch_x, batch_y = self.get_batch()

            # Reset gradients
            model.zero_grad()

            # Forward pass
            output = F.smooth_l1_loss(model(batch_x), batch_y)
            loss = output.item()

            # Backward pass
            output.backward()

            # Apply gradients
            for param in model.parameters():
                param.data.add_(-lr * param.grad.data)

            if batch_idx % 50 == 0:
                print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
        
            # Stop criterion
            if loss < 1e-3:
                break
        return (batch_idx, loss)

