import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .blocks import Conv2dBlock, GatedConv2d, IDTransposeGatedConv2d, SPADElayer, ResBlock
from .Zencoder import Zencoder

##########################################
class UnetG(nn.Module):
    def __init__(self, cfg):
        super(UnetG, self).__init__()

        input_nc = cfg['input_nc']
        ngf = cfg['ngf']
        output_nc = cfg['output_nc']
        lab_nc = 19
        g_norm = cfg['G_norm_type']
        style_dim = cfg['style_dim']

        self.Zencoder = Zencoder(output_nc=style_dim)

        # Encoder layers
        self.gen1 = Conv2dBlock(input_nc + 1, ngf, kernel_size=5, stride=1, padding=2, norm='none', activation='lrelu')
        self.gen2 = GatedConv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.gen3 = GatedConv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.gen4 = GatedConv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.gen5 = GatedConv2d(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.gen6 = GatedConv2d(ngf * 16, ngf * 16, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.gen7 = GatedConv2d(ngf * 16, ngf * 32, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.gen7_p = GatedConv2d(ngf * 32, ngf * 32, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.gen8 = GatedConv2d(ngf * 32, ngf * 32, kernel_size=3, stride=1, padding=1, dilation=1, norm=g_norm,
                                activation='lrelu')

        # Decoder layers
        self.gen9_p = IDTransposeGatedConv2d(ngf * 32 + ngf * 32, ngf * 32)
        self.gen9 = IDTransposeGatedConv2d(ngf * 32 + ngf * 16, ngf * 16)
        self.gen10 = IDTransposeGatedConv2d(ngf * 16 + ngf * 16, ngf * 8)
        # output 32x32
        self.gen11 = IDTransposeGatedConv2d(ngf * 8 + ngf * 8, ngf * 4)
        # self.Pars1 = SegBranch(ngf * 4, ngf * 4, lab_nc)
        # self.spade1 = SPADElayer(ngf * 4, style_dim)
        # output 64x64
        self.gen12 = IDTransposeGatedConv2d(ngf * 4 + ngf * 4, ngf * 2)
        self.Pars2 = SegBranch(ngf * 2, ngf * 2, lab_nc)
        self.spade2 = SPADElayer(ngf * 2, style_dim)
        # output 128x128
        self.gen13 = IDTransposeGatedConv2d(ngf * 2 + ngf * 2, ngf)
        self.Pars3 = SegBranch(ngf, ngf, lab_nc)
        self.spade3 = SPADElayer(ngf, style_dim)
        # output 256x256
        self.gen14 = IDTransposeGatedConv2d(ngf + ngf, ngf//2)
        # self.Pars4 = SegBranch(ngf//2, ngf, lab_nc)
        # self.spade4 = SPADElayer(ngf//2, style_dim)

        self.gen15 = Conv2dBlock(ngf//2, output_nc, kernel_size=3, stride=1, padding=1, norm='none', activation='tanh')

    # In this case, we have very flexible unet construction mode.
    def forward(self, input, ref_id, ref_img, ref_seg, mask, train_mode):

        z_id = ref_id
        z_id = F.normalize(z_id)
        z_id = z_id.detach()

        style_codes = self.Zencoder(ref_img, ref_seg)

        # Encoder
        # No norm on the first layer
        g1 = self.gen1(input)
        g2 = self.gen2(g1)
        g3 = self.gen3(g2)
        g4 = self.gen4(g3)
        g5 = self.gen5(g4)
        g6 = self.gen6(g5)
        g7 = self.gen7(g6)
        g7_p = self.gen7_p(g7)
        g8 = self.gen8(g7_p)
        
        g9_p = self.gen9_p(g8, z_id, skip=g7)
        g9 = self.gen9(g9_p, z_id, skip=g6)
        g10 = self.gen10(g9, z_id, skip=g5)

        # output 32x32
        g11 = self.gen11(g10, z_id, skip=g4)
        # pars1, segmap1 = self.Pars1(g11)
        # style_map_select_1, style_map_mask_1 = self.gen_style_map(style_codes, segmap1, mask)
        # g11 = self.spade1(g11, style_map_mask_1)

        # output 64x64
        g12 = self.gen12(g11, z_id, skip=g3)
        pars2, segmap2 = self.Pars2(g12)
        style_map_select_2, style_map_mask_2 = self.gen_style_map(style_codes, segmap2, mask, train_mode=train_mode)
        g12 = self.spade2(g12, style_map_mask_2)

        # output 128x128
        g13 = self.gen13(g12, z_id, skip=g2)
        pars3, segmap3 = self.Pars3(g13)
        style_map_select_3, style_map_mask_3 = self.gen_style_map(style_codes, segmap3, mask, train_mode=train_mode)
        g13 = self.spade3(g13, style_map_mask_3)

        # output 256x256
        g14 = self.gen14(g13, z_id, skip=g1)
        # pars4, segmap4 = self.Pars4(g14)
        # style_map_select_4, style_map_mask_4 = self.gen_style_map(style_codes, segmap4, mask, train_mode=train_mode)
        # g14 = self.spade4(g14, style_map_mask_4)

        # output layer
        g15 = self.gen15(g14)

        # return g15, [pars1, pars2, pars3, pars4], [segmap1, segmap2, segmap3, segmap4], \
        #        [style_map_select_1, style_map_select_2, style_map_select_3, style_map_select_4], \
        #        [style_map_mask_1, style_map_mask_2, style_map_mask_3, style_map_mask_4]

        return g15, [pars2, pars3], [ segmap2, segmap3], \
               [style_map_select_2, style_map_select_3], \
               [style_map_mask_2, style_map_mask_3]


    def gen_style_map(self, style_codes, segmap, mask, train_mode):
        bs, N, h, w = segmap.size()
        _, N, D = style_codes.size()
        style_codesT = style_codes.permute(0, 2, 1)  # B x D x N
        segmap = segmap.view(bs, N, -1)  # B x N x HW
        style_map = torch.matmul(style_codesT, segmap)
        style_map = style_map.view(bs, D, h, w)  # B x D x H x W
        style_map_select, style_map_mask = self.mask_region(segmap.view(bs, N, h, w), style_map, mask)
        
        if train_mode == 'inpainting':
            return style_map_select, style_map_mask
        elif train_mode == 'seg':
            return style_map_select, style_map_mask
        elif train_mode == 'comp':
            return style_map_select, style_map_select


    def mask_region(self, segmap, style_map, mask):
        bs, D, _, _ = style_map.size()
        _, N, h, w = segmap.size()
        mask = F.interpolate(mask, size=segmap.size()[2:], mode='nearest')
        # 过滤几个区域，只考虑几个 comp
        region_select = [2, 3, 4, 5, 12, 13]
        for j in range(N):
            if j not in region_select:
                style_map = style_map * (1.0 - segmap[:, j:j + 1])
                # print(segmap[:, j:j+1].size())  # torch.Size([12, 1, 32, 32])
                # print(segmap[:, j:j+1])
        style_map_select = style_map.clone()

        # 过滤 mask 区域
        segmap_nonmask = segmap * (1. - mask)
        for i in range(bs):
            for j in range(N):
                if j in region_select:
                    component_mask_area = torch.sum(segmap_nonmask.bool()[i, j])
                    # print(component_mask_area)
                    if component_mask_area > 0:
                        style_map[i:i + 1] = style_map[i:i + 1] * (1.0 - segmap[i:i + 1, j:j + 1])
        style_map_mask = style_map

        return style_map_select, style_map_mask


    def gen_inpainting_mask(self, segmap, mask, ref_segmap):
        bs, N, h, w = segmap.size()
        mask_init = torch.ones((bs, 1, h, w), device=segmap.device)

        # 过滤几个区域
        region_select = [2, 3, 4, 5, 12, 13]
        for j in range(N):
            if j not in region_select:
                mask_init = mask_init * (1.0 - segmap[:, j:j + 1])
                # print(segmap[:, j:j+1].size())  # torch.Size([12, 1, 32, 32])
        gen_mask_select = mask_init.clone()

        # 过滤 mask 区域
        segmap_nonmask = segmap * (1. - mask)
        for i in range(bs):
            for j in range(N):
                if j in region_select:
                    component_mask_area = torch.sum(segmap_nonmask.bool()[i, j])
                    # print(component_mask_area)
                    if component_mask_area > 0:
                        mask_init[i:i + 1] = mask_init[i:i + 1] * (1.0 - segmap[i:i + 1, j:j + 1])

                    component_mask_area_ref = torch.sum(ref_segmap.bool()[i, j])
                    if component_mask_area_ref > 0:
                        pass
                    else:
                        mask_init[i:i + 1] = mask_init[i:i + 1] * (1.0 - segmap[i:i + 1, j:j + 1])
        gen_mask = mask_init

        gen_mask_select = gen_mask_select.float().detach()
        gen_mask = gen_mask.float().detach()

        return gen_mask_select, gen_mask


############################################################################################
class SegBranch(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(SegBranch, self).__init__()
        self.n_classes = n_classes
        self.conv = Conv2dBlock(in_chan, mid_chan, kernel_size=3, stride=1, padding=1, norm='in', activation='lrelu')
        self.resblk1 = ResBlock(mid_chan)
        self.resblk2 = ResBlock(mid_chan)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        # x = x.clone().detach()
        x = self.conv(x)
        x = self.resblk1(x)
        x = self.resblk2(x)
        logit = self.conv_out(x)

        lab_map = torch.argmax(logit.detach(), dim=1, keepdim=True)
        # print(lab_map.size())  # torch.Size([8, 1, 32, 32])
        # print(lab_map)
        bs, _, h, w = lab_map.size()
        input_label = torch.cuda.FloatTensor(bs, self.n_classes, h, w).zero_()
        segmap = input_label.scatter_(1, lab_map.long(), 1.0)
        # print(segmap.size())  # torch.Size([8, 34, 32, 32])
        # print(segmap)
        return logit, segmap