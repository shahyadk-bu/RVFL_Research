import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    """
    Standard Neural Network.
    """
    def __init__(self, layer_sizes, activation=torch.relu):
        super().__init__()

        self.act = activation

        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
        ])

    def forward(self, x):

        # all hidden layers
        for layer in self.layers[:-1]:
            x = self.act(layer(x))

        # final layer
        x = self.layers[-1](x)

        return x