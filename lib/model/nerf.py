import torch
import torch.nn as nn
from lib.model.utils import get_activation_class

class NERF(nn.Module):
    """
    Input:

    """
    def __init__(
        self, input_dim, view_input_dim, num_layers=8, hidden_dim=256, actv1="relu", skip1=[4],
        feature_dim=256, num_layers_view=1, hidden_dim_view=128, actv2="relu", output_actv="sigmoid", skip2=[]
        ):
        super().__init__()    

        self.layer = nn.Linear
        self.input_dim = input_dim # dim from positional encoding of x,y,z
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip1 = skip1
        self.actv_func = get_activation_class(actv1)
        self.feature_dim = feature_dim + 1
        # self.output_dim = fea_dim + 1 # hidden dim + sigma (density)

        self.input_dim_view = feature_dim + view_input_dim # hidden dim + view_dir_encoded
        self.num_layers_view = num_layers_view
        self.hidden_dim_view = hidden_dim_view
        self.skip2 = skip2
        self.actv_func_view = get_activation_class(actv2)
        self.output_dim = 3 # rgb
        self.out_actv_func = get_activation_class(output_actv)
        
        self.make()
        
    def make(self):
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim))
            elif i in self.skip1:
                layers.append(self.layer(self.input_dim+self.hidden_dim, self.hidden_dim))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim))
        self.network1 = nn.ModuleList(layers)
        self.feature_layer = self.layer(self.hidden_dim, self.feature_dim)
        # self.sigma_layer = self.layer(self.hidden_dim, 1)
        
        layers_view = []
        for i in range(self.num_layers_view):
            if i == 0: 
                layers_view.append(self.layer(self.input_dim_view, self.hidden_dim_view))
            # elif i in self.skip2:
            #     layers_view.append(self.layer_view(self.input_dim_view+self.hidden_dim_view, self.hidden_dim_view))
            else:
                layers_view.append(self.layer(self.hidden_dim_view, self.hidden_dim_view))
        self.network2 = nn.ModuleList(layers_view)
        self.output_layer = self.layer(self.hidden_dim_view, self.output_dim)
        
    def forward(self, x, view):
        for i, layer in enumerate(self.network1):
            if i == 0:
                h = self.actv_func(layer(x))
            elif i in self.skip1:
                h = torch.cat([x, h], dim=-1)
                h = self.actv_func(layer(h))
            else:
                h = self.actv_func(layer(h))
        
        # no actv_func for feature layer
        feature = self.feature_layer(h)
        sigma = feature[..., :1]

        x = torch.cat([feature[..., 1:], view], dim=-1)
        
        for i, layer in enumerate(self.network2):
            if i == 0:
                h = self.actv_func_view(layer(x))
            # elif i in self.skip2:
            #     h = torch.cat([x, h], dim=-1)
            #     h = self.actv_func_view(layer(h))
            else:
                h = self.actv_func_view(layer(h))

        rgb = self.out_actv_func(self.output_layer(h))
        
        return rgb, torch.nn.Softplus(beta=100)(sigma)
