"""
Torch class for creating feedforward networks, specifically Q-value networks.

You do *NOT* need to modify the code here.
"""
from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Q-value network for DQN, parameterized as a feedforward NN.
    Takes in an input of size (state,) and outputs (action,).
    """

    def __init__(self, input_dim, output_dim, hidden_dim=256):
        """
        Initialize a new instance of Q-value network. One hidden layer
        with 256 hidden units.

        Args:
          input_dim (int): input dimension
          output_dim (int): output dimension
        """
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        """
        Forward pass for the NN.
        input-> linear-> ReLU -> linear -> output.

        Args:
          inputs (torch.tensor): torch tensor of size (batch_size, input_dim)

        Returns:
          (torch.tensor): torch tensor of size (batch_size, output_dim)
        """
        x = F.relu(self.dense1(inputs))
        return self.out(x)
