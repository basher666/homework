import torch
import numpy as np
import torch.nn as nn

class policy(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(policy, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h1_relu = nn.functional.relu(self.linear1(x))
        h2_relu = nn.functional.relu(self.linear2(h1_relu))
        y_pred = self.linear3(h2_relu)
        return y_pred