import torch
import torch.nn as nn
import torch.utils.checkpoint
from prettytable import PrettyTable


class Layer(nn.Sequential):
  def __init__(self, *args, **kwargs):
      super(Layer,self).__init__(*args)

  def forward(self, *args):
    return super().forward(*args)


class CheckPointed(nn.Sequential):
  def __init__(self, *args, **kwargs):
      super(CheckPointed, self).__init__(*args)
  
  def forward(self, *args):
    return torch.utils.checkpoint.checkpoint(super().forward, *args)

class BCCModel(nn.Module):
  def __init__(self) -> None:
    super(BCCModel,self).__init__()
    pass

  @property
  def size(self):
    return self.__class__
  
  def forward(self, *args):
     raise NotImplementedError

  def params(self):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in self.__class__.named_parameters(self):
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

  
  @property
  def name(self):
    raise NotImplementedError