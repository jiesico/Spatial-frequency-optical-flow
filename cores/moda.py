import torch.nn as nn
import torch
from visdom import Visdom

class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        #print(len(bn))
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
       
        bn1_list = sorted(bn1)
        bn2_list = sorted(bn2)
        bn_threshold1 = bn1_list[int(len(bn1)/4)]
        bn_threshold2 = bn2_list[int(len(bn2)/4)]

        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold1] = x[0][:, bn1 >= bn_threshold1]
        x1[:, bn1 < bn_threshold1] = x[1][:, bn1 < bn_threshold1]
        x2[:, bn2 >= bn_threshold2] = x[1][:, bn2 >= bn_threshold2]
        x2[:, bn2 < bn_threshold2] = x[0][:, bn2 < bn_threshold2]
        #self.total_steps = self.total_steps + 1
        return [x1, x2]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class InstanceNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(InstanceNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'insnorm_' + str(i), nn.InstanceNorm2d(num_features,affine=True, track_running_stats=True))

    def forward(self, x_parallel):
        return [getattr(self, 'insnorm_' + str(i))(x) for i, x in enumerate(x_parallel)]


