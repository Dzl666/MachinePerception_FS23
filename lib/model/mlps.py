import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.utils import get_activation_class

class MLP(nn.Module):
    """Super basic but super useful MLP class.
    """
    def __init__(self, input_dim, output_dim, activation = "relu", bias = True,
        layer = nn.Linear, num_layers = 4, hidden_dim = 128, skip = [2]):
        """Initialize the MLP.
        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.activation = get_activation_class(activation)
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        
        self.make()

    def make(self):
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)

    def forward(self, x, return_h=False):
        """
        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.
        Returns: (torch.FloatTensor, (optional) torch.FloatTensor):
            - The output tensor of shape [batch, ..., output_dim]
            - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = torch.cat([x, h], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)
        
        if return_h:
            return out, h
        else:
            return out
