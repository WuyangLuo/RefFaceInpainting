import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_norm_layer
import functools
import numpy as np


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, cfg, input_nc):
        super().__init__()
        self.cfg = cfg
        self.input_nc = input_nc

        for i in range(self.cfg['num_D']):
            subnetD = self.create_single_discriminator(self.cfg)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, cfg):
        netD = NLayerD(cfg, self.input_nc)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.cfg['no_ganFeat_loss']
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerD(nn.Module):
    def __init__(self, cfg, input_nc):
        super().__init__()
        self.cfg = cfg

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = cfg['ndf']
        # input_nc = self.compute_D_input_nc(self.cfg)

        D_norm_type = cfg['D_norm_type']
        norm_layer = get_norm_layer(norm_type=D_norm_type)

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, cfg['n_layers_D']):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == cfg['n_layers_D'] - 1 else 2
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw, bias=use_bias),
                          norm_layer(nf),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    # def compute_D_input_nc(self, cfg):
        # input_nc = cfg['lab_dim'] + cfg['output_nc']
        # return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.cfg['no_ganFeat_loss']
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]