import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from .blocks import Conv2dBlock, UpConv2dBlock

class ComponentD(nn.Module):
    def __init__(self, cfg):
        super(ComponentD, self).__init__()

        input_nc = cfg['input_nc']
        ndf = cfg['ndf']

        self.dis1 = Conv2dBlock(input_nc, ndf, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')
        self.dis2 = Conv2dBlock(ndf, ndf * 2, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')
        self.dis3 = Conv2dBlock(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')
        self.dis4 = Conv2dBlock(ndf * 4, 1, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')

    def forward(self, input):
        d1 = self.dis1(input)
        d2 = self.dis2(d1)
        d3 = self.dis3(d2)
        d4 = self.dis4(d3)

        return d4