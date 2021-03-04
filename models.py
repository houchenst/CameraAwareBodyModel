from torch import nn
import torch

class Unstructured3D(nn.Module):

    def __init__(self, dimensions=3, hidden_size=40, num_layers=4):
        super().__init__()
        # store hyperparameters
        self.dimensions = 3
        self.hidden_size = 40
        self.num_layers = 4

        # activation function
        self.lr = nn.LeakyReLU(0.1)
        self.sig = nn.Sigmoid()

        # model parameters
        assert(num_layers >= 2)
        first_layer = nn.Linear(dimensions, hidden_size)
        last_layer = nn.Linear(hidden_size, 1)
        middle_layers = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-2)]
        self.linear_layers = nn.ModuleList([first_layer] + middle_layers + [last_layer])
        
    
    def forward(self, query_points):
        output = query_points

        # pass through each layer
        for i in range(self.num_layers):
            output = self.linear_layers[i](output)
            # LeakyReLU activation
            if i < self.num_layers - 1:
                output = self.lr(output)
            # Sigmoid for last layer
            else:
                output = self.sig(output)
        
        return output
    