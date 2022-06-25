from functools import partial
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from models import ActivationLinear
from utils import eye_like

class FOOFWraper():
  moving_yy = dict()
  damped_invers_yy = dict()

  def __init__(
    self, model: torch.nn.Module, learning_rate: float, damping: float, inverse_step: int=1, moving_step: int=1, m: float=0.95
):
    assert moving_step <= inverse_step

    self.model = model

    self.learning_rate = learning_rate
    self.damping = damping
    self.inverse_step = inverse_step
    self.moving_step = moving_step
    self.m = m

    self.iter_step = 0

    self._prepare_foof()

  def __call__(self, x):
    return self.model(x)

  def _prepare_foof(self):
    for name, m in self.model.named_modules():
      if isinstance(m, ActivationLinear):
        m.register_forward_hook(partial(self._update_yy_inv, name))

  @torch.no_grad()
  def _update_yy_inv(self, name, mod, inp, out):
    inp = inp[0]
    if mod.linear.bias is not None:
      inp = F.pad(inp, (0, 1), mode='constant', value=1.0)

    if (self.iter_step % self.moving_step) == 0:
      if self.iter_step == 0:
        self.moving_yy[name] = torch.matmul(inp.T, inp)
      else:
        self.moving_yy[name] = self.m * self.moving_yy[name] + (1 - self.m) * torch.matmul(inp.T, inp)

    if (self.iter_step % self.inverse_step) == 0:
      self.damped_invers_yy[name] = torch.inverse(self.damping * eye_like(self.moving_yy[name]) + self.moving_yy[name])

  def step(self):
    self.iter_step += 1
    self._foof_step()

  @torch.no_grad()
  def _foof_step(self):
    for name, m in self.model.named_modules():
      if isinstance(m, ActivationLinear):
        w = m.linear.weight.data
        w_grad = m.linear.weight.grad

        if m.linear.bias is not None:  # concat biase and weight matrix
          w = torch.cat((w, m.linear.bias.data.unsqueeze(1)), 1)
          w_grad = torch.cat((w_grad, m.linear.bias.grad.unsqueeze(1)), 1)

        w = w - self.learning_rate * torch.matmul(w_grad, self.damped_invers_yy[name])

        if m.linear.bias is not None:
          m.linear.weight.data[:] = w[:, :-1]
          m.linear.bias.data[:] = w[:, -1]
        else:
          m.linear.weight.data[:] = w


class LocoPropWraper():
  a_linear_y = dict()

  def __init__(self, model: torch.nn.Module, lambda_a: float, lambda_r: float, iner_solver: Optional[Callable], f_type: str='identity', **iner_kwargs):
    assert f_type in ['identity', 'activation'], "type should be 'identity' or 'activation'"
    assert lambda_a > 0, 'lambda_a should be > 0'
    assert lambda_r > 0, 'lambda_r should be > 0'

    self.model = model
    self.lambda_a = lambda_a
    self.lambda_r = lambda_r
    self.f_type = f_type
    
    self.iner_solver = iner_solver
    self.iner_kwargs = iner_kwargs

    self._prepare_loco_prop()
  
  def __call__(self, x):
    return self.model(x)

  def _prepare_loco_prop(self):
    for name, m in self.model.named_modules():
      if isinstance(m, ActivationLinear):
        m.linear.register_forward_hook(partial(self._save_a_linear_y, name, m.f))

  def _save_a_linear_y(self, name, f, mod, inp, out):
    out.retain_grad()
    self.a_linear_y[name] = (out, f, mod, inp[0])
  
  def step(self):
    self._loco_prop_step()

  @torch.no_grad()
  def _loco_prop_step(self):
    for name, (a, f, linear, y) in self.a_linear_y.items():
      if self.f_type == 'identity':  # S type: f(x) = x
        f = lambda x: x
      elif self.f_type == 'activation':  # M type: nonlinearity after Linear layer
        pass

      y_new = f(a) - self.lambda_a * a.grad  # ak f(x) = x -> y_new == a_new
      
      if linear.bias is not None:  # concat biase and weight matrix
        w = torch.cat((linear.weight.data, linear.bias.data.unsqueeze(1)), 1)
        y = F.pad(y, (0, 1), mode='constant', value=1.0)
      else:
        w = linear.weight.data

      f_iner = lambda w_new: w - (1/self.lambda_r) * torch.matmul(torch.transpose(f(torch.matmul(y, w_new.T)) - y_new, -1, -2), y)
      w = self.iner_solver(f=f_iner, x0=w, **self.iner_kwargs)
      
      if linear.bias is not None:
        linear.weight.data[:] = w[:, :-1]
        linear.bias.data[:] = w[:, -1]
      else:
         linear.weight.data[:] = w