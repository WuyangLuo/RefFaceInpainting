import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Zencoder(torch.nn.Module):
    def __init__(self, input_nc=3, output_nc=256, ngf=32, n_downsampling=2, norm_layer=nn.InstanceNorm2d):
        super(Zencoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0),
                 norm_layer(ngf), nn.LeakyReLU(0.2, False)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, False)]

        ### upsample
        for i in range(1):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.LeakyReLU(0.2, False)]

        model += [nn.ReflectionPad2d(1), nn.Conv2d(256, output_nc, kernel_size=3, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)


    def forward(self, input, segmap):

        codes = self.model(input)

        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')

        b_size = codes.shape[0]
        f_size = codes.shape[1]
        s_size = segmap.shape[1]

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)


        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size, component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

        return codes_vector