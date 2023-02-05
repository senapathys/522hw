import torch
from typing import Callable
import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    """
    Multi-layer perceptron class implementation
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_classes: int,
            hidden_count: int = 1,
            activation: Callable = torch.nn.ReLU,
            initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        # Initialize the hidden layers
        hidden_layers = []
        for i in range(hidden_count):
            hidden_layers.append(nn.Linear(input_size, hidden_size))
            hidden_layers.append(activation())
            input_size = hidden_size

        # Initialize the output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

        # Stack the hidden layers and output layer into the model
        self.model = nn.Sequential(*hidden_layers, self.output_layer)

        # Initialize the weights
        for layer in self.model:
            if hasattr(layer, "weight"):
                initializer(layer.weight)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        return self.model(x)
