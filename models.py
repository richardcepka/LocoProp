import torch
from torch import nn

class ActivationLinear(nn.Module):
  def __init__(self, in_features: int, out_features: int, bias: bool=True, f=lambda x: x):
     super(ActivationLinear, self).__init__()
     self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
     self.f = f
  
  def forward(self, x):
    return self.f(self.linear(x))


class MLP(nn.Module):
  def __init__(self, in_dim: int, hid_dim: int, out_dim: int, n_hid_layers: int, nonlinearity=torch.tanh, bias: bool=True):
        super(MLP, self).__init__()

        self.f_in = ActivationLinear(in_dim, hid_dim, bias=bias, f=nonlinearity)
        self.layers = nn.ModuleList([ActivationLinear(hid_dim, hid_dim, bias=bias, f=nonlinearity) for _ in range(n_hid_layers)])
        self.f_out = ActivationLinear(hid_dim, out_dim, bias=bias)
  
  def forward(self, x):
    x = self.f_in(x)
    for f in self.layers:
      x = f(x)
    return self.f_out(x)