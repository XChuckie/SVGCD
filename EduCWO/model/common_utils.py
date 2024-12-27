import torch
import torch.nn as nn
import torch.nn.functional as F


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()
        
    def __call__(self, module):
        if hasattr(module, 'weight'):
            weight = module.weight.data
            a = F.relu(torch.neg(weight))
            weight.add_(a)