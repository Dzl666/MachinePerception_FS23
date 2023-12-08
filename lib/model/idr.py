import torch
import torch.nn as nn
from lib.model.utils import get_activation_class

class ImplicitDifferentiableRenderer(nn.Module):
    """
    Input:

    """
    def __init__(
        self, input_dim, input_dim2, num_layers=8, hidden_dim=256, actv1="relu", skip1=[4],
        feature_dim=256, num_layers2=4, hidden_dim2=256, actv2="relu", skip2=[], output_actv="sigmoid"
        ):
        super().__init__()    
        self.layer = nn.Linear

        self.input_dim = input_dim # dim from positional encoding of x,y,z
        self.num_layers_sdf = num_layers
        self.hidden_dim = hidden_dim
        self.skip1 = skip1
        self.actv_func = get_activation_class(actv1)
        self.output_dim = feature_dim + 1 # feature + sigma

        # coord_encoded + view_encoded + hidden dim + grad
        self.input_dim2 = input_dim + input_dim2 + feature_dim + input_dim
        self.num_layers_render = num_layers2
        self.hidden_dim2 = hidden_dim2
        self.skip2 = skip2
        self.actv_func2 = get_activation_class(actv2)
        self.output_dim2 = 3 # rgb
        self.out_actv_func = get_activation_class(output_actv)
        
        self.make()
        
    def make(self):
        layers = []
        for i in range(self.num_layers_sdf):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim))
            elif i in self.skip1:
                layers.append(self.layer(self.input_dim+self.hidden_dim, self.hidden_dim))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim))
        self.network_sdf = nn.ModuleList(layers)
        self.output_layer = self.layer(self.hidden_dim, self.output_dim)
        
        layers = []
        for i in range(self.num_layers_render):
            if i == 0: 
                layers.append(self.layer(self.input_dim2, self.hidden_dim2))
            elif i in self.skip2:
                layers.append(self.layer(self.input_dim2+self.hidden_dim2, self.hidden_dim_view))
            else:
                layers.append(self.layer(self.hidden_dim2, self.hidden_dim2))
        self.network_render = nn.ModuleList(layers)
        self.output_layer2 = self.layer(self.hidden_dim2, self.output_dim2)
    
    def forward_sdf(self, x):
        for i, layer in enumerate(self.network_sdf):
            if i == 0:
                h = self.actv_func(layer(x))
            elif i in self.skip1:
                h = torch.cat([x, h], dim=-1)
                h = self.actv_func(layer(h))
            else:
                h = self.actv_func(layer(h))
        output = self.output_layer(h)
        return output
    
    def forward_grad_sample(self, x):
        x.requires_grad_(True)
        y = self.sdf_net(x)[...,0]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        y.backward(gradient=d_output)
        return x.grad

    def forward_render(self, x):
        for i, layer in enumerate(self.network_render):
            if i == 0:
                h = self.actv_func2(layer(x))
            elif i in self.skip2:
                h = torch.cat([x, h], dim=-1)
                h = self.actv_func2(layer(h))
            else:
                h = self.actv_func2(layer(h))
        output = self.out_actv_func(self.output_layer2(h))
        return output

    def forward(self, x, view):

        grad = self.forward_grad_sample(x)
        output_sdf = self.network_sdf(x)

        sigma = self.actv_func(output_sdf[..., :1])
        feature = output_sdf[..., 1:]

        rgb = self.forward_render(torch.cat([x, view, grad, feature], dim=-1))
    
        return rgb, sigma
