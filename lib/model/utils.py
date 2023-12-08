import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation_class(activation_type):
    """Utility function to return an activation function class based on the string description.
    Args:
        activation_type (str): The name for the activation function.
    Returns:
        (Function): The activation function to be used. 
    """
    if activation_type == 'relu':
        return torch.relu
    elif activation_type == 'lrelu':
        return torch.nn.LeakyReLU()
    elif activation_type == 'sin':
        return torch.sin
    elif activation_type == 'sigmoid':
        return torch.sigmoid
    elif activation_type == 'softplus':
        return torch.nn.Softplus(beta=100, threshold=20)
    else:
        assert False and "activation type does not exist"
