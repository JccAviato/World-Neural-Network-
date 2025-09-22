import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[64,128,64], dropout=0.1, is_regression=False):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(p=dropout)]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)
        self.is_regression = is_regression

    def forward(self, x):
        out = self.net(x)
        if not self.is_regression:
            return out
        return out.squeeze(-1)

class CountryFromLatLonNN(MLP):
    def __init__(self, n_countries, in_dim=2):
        super().__init__(in_dim=in_dim, out_dim=n_countries, hidden=[128,128,64], dropout=0.15, is_regression=False)

class ContinentFromLatLonNN(MLP):
    def __init__(self, n_continents, in_dim=2):
        super().__init__(in_dim=in_dim, out_dim=n_continents, hidden=[64,128,64], dropout=0.1, is_regression=False)

class GreatCircleDistanceNN(MLP):
    def __init__(self, in_dim=4):
        super().__init__(in_dim=in_dim, out_dim=1, hidden=[128,128,64], dropout=0.1, is_regression=True)
