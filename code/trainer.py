from networks.UnetG import UnetG
from networks.discriminator import MultiscaleDiscriminator
from networks.SNPatchDiscriminator import SNPatchDiscriminator
from networks.compD import ComponentD
from utils import get_scheduler, weights_init, save_network, save_latest_network, get_model_list
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import loss
import os

from networks.arcface_models import resnet18, resnet34, resnet50, resnet101
from parsing.model import BiSeNet


class BiMod_Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # setting basic params
        self.cfg = cfg
        if self.cfg['is_train']:
            self.model_names = ['netG', 'netD_Glbal', 'netD_le', 'netD_re', 'netD_mouth']
            # self.model_names = ['netG', 'netD_Glbal', 'netD_Glbal_comp']
        else:  # during test time, only load G
            self.model_names = ['netG']

        # =============================================================================================================
        # arcface
        self.arcface = resnet101()
        self.arcface.load_state_dict(torch.load('/home/lwy/research/inpaintingID/pretrained/BEST_checkpoint_r101.pth'))
        print('load arcface')
        self.set_requires_grad(self.arcface, False)

        # # parsing model
        # self.parsingnet = BiSeNet(n_classes=19)
        # self.parsingnet.load_state_dict(torch.load('parsing/79999_iter.pth'))
        # print('load parsingnet')
        # self.set_requires_grad(self.parsingnet, False)
        # =============================================================================================================

        # Initiate the submodules and initialization params
        self.netG = UnetG(self.cfg)
        self.netD_Glbal = SNPatchDiscriminator(self.cfg)
        # self.netD_Glbal_comp = SNPatchDiscriminator(self.cfg)
        # self.netD_patch = MultiscaleDiscriminator(self.cfg, self.cfg['output_nc'])
        self.netD_le = ComponentD(self.cfg)
        self.netD_re = ComponentD(self.cfg)
        self.netD_mouth = ComponentD(self.cfg)

        self.netG.apply(weights_init('gaussian'))
        self.netD_Glbal.apply(weights_init('gaussian'))
        # self.netD_Glbal_comp.apply(weights_init('gaussian'))
        # self.netD_patch.apply(weights_init('gaussian'))
        self.netD_le.apply(weights_init('gaussian'))
        self.netD_re.apply(weights_init('gaussian'))
        self.netD_mouth.apply(weights_init('gaussian'))

        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() \
            else torch.ByteTensor

        # Setup the optimizers and schedulers
        if cfg['is_train']:
            lr = self.cfg['lr']
            beta1 = self.cfg['beta1']
            beta2 = self.cfg['beta2']
            # set optimizers
            self.optimizers = []
            G_params = list(self.netG.parameters())
            D_global_params = list(self.netD_Glbal.parameters())
            # D_global_comp_params = list(self.netD_Glbal_comp.parameters())
            # D_patch_params = list(self.netD_patch.parameters())
            D_le_params = list(self.netD_le.parameters())
            D_re_params = list(self.netD_re.parameters())
            D_mouth_params = list(self.netD_mouth.parameters())

            self.optimizer_G = torch.optim.Adam(G_params, lr=lr, betas=(beta1, beta2))
            self.optimizer_D_global = torch.optim.Adam(D_global_params, lr=lr, betas=(beta1, beta2))
            # self.optimizer_D_global_comp = torch.optim.Adam(D_global_comp_params, lr=lr, betas=(beta1, beta2))
            # self.optimizer_D_patch = torch.optim.Adam(D_patch_params, lr=lr, betas=(beta1, beta2))
            self.optimizer_D_le = torch.optim.Adam(D_le_params, lr=lr, betas=(beta1, beta2))
            self.optimizer_D_re = torch.optim.Adam(D_re_params, lr=lr, betas=(beta1, beta2))
            self.optimizer_D_mouth = torch.optim.Adam(D_mouth_params, lr=lr, betas=(beta1, beta2))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_global)
            # self.optimizers.append(self.optimizer_D_global_comp)
            # self.optimizers.append(self.optimizer_D_patch)
            self.optimizers.append(self.optimizer_D_le)
            self.optimizers.append(self.optimizer_D_re)
            self.optimizers.append(self.optimizer_D_mouth)
            
            self.opt_names = ['optimizer_G', 'optimizer_D_global', 'optimizer_D_le', 'optimizer_D_re', 'optimizer_D_mouth']

            # set schedulers
            self.schedulers = [get_scheduler(optimizer, self.cfg) for optimizer in self.optimizers]
            # set criterion
            self.criterionGAN_global = loss.GANLoss(cfg['gan_mode']).cuda()
            self.criterionGAN_patch = loss.GANLoss_MultiD(cfg['gan_mode'], tensor=self.FloatTensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = loss.VGGLoss()
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionArcface = loss.Arcface_Loss(self.arcface)
            self.criterionGAN_comp = loss.GANLoss(cfg['gan_mode']).cuda()
            self.criterionParsing = loss.Parsing_Loss()

        self.G_losses = {}
        self.D_global_losses = {}
        self.D_patch_losses = {}
        self.D_comp_losses = {}

        self.lambda_L1 = self.cfg['lambda_L1']
        self.G_msParsing = 1.0
        self.lambda_arcface = self.cfg['lambda_arcface']
        self.lambda_vgg = self.cfg['lambda_vgg']

    ######################################################################################
    def set_input(self, input):
        # scatter_ require .long() type
        self.masked_img = input['masked_img'].cuda()     # mask image
        self.gt = input['img_corr'].cuda()        # real image
        self.ref = input['img_ref'].cuda()
        self.mask = input['mask'].cuda()    # mask image
        self.lab_gt = input['lab_gt'].cuda()
        self.lab_ref = input['lab_ref'].cuda()
        self.name = input['name']

        self.gt_segmap = self.gen_segmap(self.lab_gt)
        self.ref_segmap = self.gen_segmap(self.lab_ref)
        self.pars_gt = input['pars_gt'].cuda()
        self.pars_ref = input['pars_ref'].cuda()

        self.mask_seg = input['mask_seg'].cuda()

        self.gt_le_bbox = input['gt_le']
        self.gt_re_bbox = input['gt_re']
        self.gt_nose_bbox = input['gt_nose']
        self.gt_mouth_bbox = input['gt_mouth']
        
        self.ref_le_bbox = input['ref_le']
        self.ref_re_bbox = input['ref_re']
        self.ref_nose_bbox = input['ref_nose']
        self.ref_mouth_bbox = input['ref_mouth']

    def gen_segmap(self, lab):
        # create one-hot label map
        lab_map = lab
        bs, _, h, w = lab_map.size()
        nc = 19
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        segmap = input_label.scatter_(1, lab_map.long(), 1.0)
        return segmap

    def forward(self, train_mode):
        ref_id = self.arcface(F.interpolate(self.ref, size=112, mode='bilinear'))
        gt_id = self.arcface(F.interpolate(self.gt, size=112, mode='bilinear'))

        if train_mode == 'inpainting':
            self.fake, pars_list, segmap_list, style_map_select_list, style_map_mask_list = \
                    self.netG(torch.cat((self.masked_img, self.mask), dim=1), ref_id, self.ref, self.ref_segmap, self.mask, train_mode=train_mode)
            self.masked_fake = self.fake * self.mask + self.gt * (1. - self.mask)

        elif train_mode == 'comp':
            self.mask = self.gen_inpainting_mask(self.gt_segmap)
            self.masked_img = self.gt * (1. - self.mask)
            self.fake, pars_list, segmap_list, style_map_select_list, style_map_mask_list = \
                    self.netG(torch.cat((self.masked_img, self.mask), dim=1), gt_id, self.gt, self.gt_segmap, self.mask, train_mode=train_mode)
            self.masked_fake = self.fake * self.mask + self.gt * (1. - self.mask)

        elif train_mode == 'seg':
            self.mask = self.mask_seg * (1. - self.mask)
            self.masked_img = self.gt * (1. - self.mask)
            self.fake, pars_list, segmap_list, style_map_select_list, style_map_mask_list = \
                    self.netG(torch.cat((self.masked_img, self.mask), dim=1), ref_id, self.gt, self.gt_segmap, self.mask, train_mode=train_mode)
            self.masked_fake = self.fake * self.mask + self.gt * (1. - self.mask)

        # unpack
        self.pars2, self.pars3 = pars_list
        self.segmap2, self.segmap3 = segmap_list

        self.style_map_select_2, self.style_map_select_3 = style_map_select_list
        self.style_map_mask_2, self.style_map_mask_3 = style_map_mask_list

        self.comp_crop()

        # i = 0
        # for name,parameters in self.netG.Zencoder.named_parameters():
            # if i == 0:
                # print(name,':',parameters[0,0,0,0])
                # print(name, ':', parameters.size())
            # i = i + 1

        # i = 0
        # for name,parameters in self.netG.named_parameters():
            # if i == 0:
                # print(name,':',parameters[0,0,0,0])
                # print(name, ':', parameters.size())
            # i = i + 1

    def forward_test(self):
        ref_id = self.arcface(F.interpolate(self.ref, size=112, mode='bilinear'))

        self.fake, pars_list, segmap_list, style_map_select_list, style_map_mask_list = \
            self.netG(torch.cat((self.masked_img, self.mask), dim=1), ref_id, self.ref, self.ref_segmap, self.mask, train_mode='inpainting')
        self.masked_fake = self.fake * self.mask + self.gt * (1. - self.mask)


    def test(self, gt, masked_img, ref, mask, ref_label):
        with torch.no_grad():
            ref_segmap = self.gen_segmap(ref_label)
            ref_id = self.arcface(F.interpolate(ref, size=112, mode='bilinear'))

            fake, _, _, _, _ = self.netG(torch.cat((masked_img, mask), dim=1), ref_id, ref, ref_segmap, mask, train_mode='inpainting')
            masked_fake = fake * mask + gt * (1. - mask)
            
            return masked_fake


    def test_split_id_style(self, gt, masked_img, ref_id, ref_style, mask, ref_label):
        with torch.no_grad():
            ref_segmap = self.gen_segmap(ref_label)
            ref_id = self.arcface(F.interpolate(ref_id, size=112, mode='bilinear'))

            fake, _, _, _, _ = self.netG(torch.cat((masked_img, mask), dim=1), ref_id, ref_style, ref_segmap, mask, train_mode='inpainting')
            masked_fake = fake * mask + gt * (1. - mask)
            
            return masked_fake


    def test_multi_ref(self, input):
        with torch.no_grad():
            masked_img = input['masked_img'].cuda()  # mask image
            gt = input['img_corr'].cuda()  # real image
            ref = input['img_ref'].cuda()
            mask = input['mask'].cuda()  # mask image
            
            lab_ref = input['lab_ref'].cuda()
            ref_segmap = self.gen_segmap(lab_ref)

            b = gt.size()[0]

            out_fake = []
            out_masked_img = []
            out_ref = []
            out_gt = []

            with torch.no_grad():
                for i_corr in range(b):
                    for i_ref in range(b):
                        # ref_id = self.arcface(F.interpolate(ref[i_ref:i_ref + 1], size=112, mode='bilinear'))
                        # fake, masks = self.netG(torch.cat((masked_img[i_corr:i_corr + 1], mask[i_corr:i_corr + 1]), dim=1),
                                                # ref_id)
                        # masked_fake = fake * mask[i_corr:i_corr + 1] + gt[i_corr:i_corr + 1] * (
                                    # 1. - mask[i_corr:i_corr + 1])

                        ref_id = self.arcface(F.interpolate(ref[i_ref:i_ref + 1], size=112, mode='bilinear'))

                        fake, _, _, _, _ = \
                            self.netG(torch.cat((masked_img[i_corr:i_corr + 1], mask[i_corr:i_corr + 1]), dim=1), ref_id, ref[i_ref:i_ref + 1],
                                      ref_segmap[i_ref:i_ref + 1], mask[i_corr:i_corr + 1], train_mode='inpainting')
                        masked_fake = fake * mask[i_corr:i_corr + 1] + gt[i_corr:i_corr + 1] * (1. - mask[i_corr:i_corr + 1])

                        out_fake.append(masked_fake)
                        out_masked_img.append(masked_img[i_corr:i_corr + 1])
                        out_ref.append(ref[i_ref:i_ref + 1])
                        out_gt.append(gt[i_corr:i_corr + 1])

        return out_masked_img, out_ref, out_gt, out_fake

    def preprocessing(self, x):
        self.mean = torch.tensor((0.485, 0.456, 0.406)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        self.std = torch.tensor((0.229, 0.224, 0.225)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        x = (x + 1.0) / 2.0
        x = F.interpolate(x, size=512, mode='bilinear')
        x = (x - self.mean) / self.std
        return x

    def logit_to_segmap(self, logit, size=(256, 256), n_classes=19):
        lab_map = torch.argmax(logit.detach(), dim=1, keepdim=True)
        bs, _, h, w = lab_map.size()
        input_label = torch.cuda.FloatTensor(bs, n_classes, h, w).zero_()
        segmap = input_label.scatter_(1, lab_map.long(), 1.0)
        segmap = F.interpolate(segmap, size=size, mode='nearest')

        return segmap

    def gen_inpainting_mask(self, segmap):
        bs, N, h, w = segmap.size()
        mask_init = torch.ones((bs, 1, h, w), device=segmap.device)

        # 过滤几个区域
        region_select = [2, 3, 4, 5, 12, 13]
        for j in range(N):
            if j not in region_select:
                mask_init = mask_init * (1.0 - segmap[:, j:j + 1])
                # print(segmap[:, j:j+1].size())  # torch.Size([12, 1, 32, 32])
        gen_mask_select = mask_init.clone()

        gen_mask_comp = gen_mask_select.float().detach()

        gen_mask_comp_dilate = self.dilate_mask(gen_mask_comp)

        return gen_mask_comp_dilate

    def dilate_mask(self, mask):
        input = mask
        b, _, _, _ = mask.size()
        mask = mask.cpu().float().numpy().astype(np.uint8)
        dilate_mask = []
        for n in range(b):
            curr_mask = mask[n, 0, :, :]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            curr_dilate_mask = cv2.dilate(curr_mask, kernel)
            curr_dilate_mask = curr_dilate_mask.reshape((1, 1,) + curr_dilate_mask.shape).astype(np.float32)
            curr_dilate_mask = torch.tensor(curr_dilate_mask, dtype=input.dtype, device=input.device, requires_grad=False)
            dilate_mask.append(curr_dilate_mask)

        dilate_mask = torch.cat(dilate_mask, dim=0)

        return dilate_mask.float().detach()

    # def test(self):
        # with torch.no_grad():
            # self.forward()

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.
    def discriminate(self, Dnet, feak, real, for_dis):
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        if not for_dis:
            fake_and_real = torch.cat([feak, real], dim=0)
        else:
            fake_and_real = torch.cat([feak.detach(), real], dim=0)

        discriminator_out = Dnet(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def compute_D_comp(self):
        """Calculate losses the discriminator"""
        fake_concat = self.fake_le
        real_concat = self.gt_le
        pred_fake, pred_real = self.discriminate(self.netD_le, fake_concat, real_concat, for_dis=True)
        self.D_comp_losses['D_fake_le'] = self.criterionGAN_comp(pred_fake, False, for_discriminator=True) * self.cfg['lambda_comp']
        self.D_comp_losses['D_real_le'] = self.criterionGAN_comp(pred_real, True, for_discriminator=True) * self.cfg['lambda_comp']

        fake_concat = self.fake_re
        real_concat = self.gt_re
        pred_fake, pred_real = self.discriminate(self.netD_re, fake_concat, real_concat, for_dis=True)
        self.D_comp_losses['D_fake_re'] = self.criterionGAN_comp(pred_fake, False, for_discriminator=True) * self.cfg['lambda_comp']
        self.D_comp_losses['D_real_re'] = self.criterionGAN_comp(pred_real, True, for_discriminator=True) * self.cfg['lambda_comp']

        fake_concat = self.fake_mouth
        real_concat = self.gt_mouth
        pred_fake, pred_real = self.discriminate(self.netD_mouth, fake_concat, real_concat, for_dis=True)
        self.D_comp_losses['D_fake_mouth'] = self.criterionGAN_comp(pred_fake, False, for_discriminator=True) * self.cfg['lambda_comp']
        self.D_comp_losses['D_real_mouth'] = self.criterionGAN_comp(pred_real, True, for_discriminator=True) * self.cfg['lambda_comp']

        return self.D_comp_losses

    # def compute_D_patch_loss(self):
        # """Calculate losses the discriminator"""

        # fake_concat = self.fake * self.mask
        # real_concat = self.gt * self.mask
        # pred_fake, pred_real = self.discriminate(self.netD_patch, fake_concat, real_concat, for_dis=True)

        # self.D_patch_losses['D_patch_fake'] = self.criterionGAN_patch(pred_fake, False, for_discriminator=True)
        # self.D_patch_losses['D_patch_real'] = self.criterionGAN_patch(pred_real, True, for_discriminator=True)

        # return self.D_patch_losses


    def compute_D_global_loss(self):
        """Calculate GAN loss for the discriminator"""
        # Fake
        fake = self.masked_fake.detach()
        pred_fake = self.netD_Glbal(fake)
        self.D_global_losses['loss_D_global_fake'] = self.criterionGAN_global(pred_fake, False, for_discriminator=True)
        # Real
        real = self.gt
        pred_real = self.netD_Glbal(real)
        self.D_global_losses['loss_D_global_real'] = self.criterionGAN_global(pred_real, True, for_discriminator=True)

        return self.D_global_losses

    # def compute_D_global_comp_loss(self):
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake
    #     fake = self.masked_fake.detach()
    #     pred_fake = self.netD_Glbal_comp(fake)
    #     self.D_global_losses['loss_D_global_fake'] = self.criterionGAN_global(pred_fake, False, for_discriminator=True)
    #     # Real
    #     real = self.gt
    #     pred_real = self.netD_Glbal_comp(real)
    #     self.D_global_losses['loss_D_global_real'] = self.criterionGAN_global(pred_real, True, for_discriminator=True)
    #
    #     return self.D_global_losses


    def compute_G_loss(self, train_mode):
        """Calculate losses for the generator"""
        # # GAN loss patch
        # fake_concat = self.fake * self.mask
        # real_concat = self.gt * self.mask
        # pred_fake_patch, pred_real_patch = self.discriminate(self.netD_patch, fake_concat, real_concat, for_dis=False)
        # self.G_losses['G_GAN_patch'] = self.criterionGAN_patch(pred_fake_patch, True, for_discriminator=False)

        if train_mode == 'inpainting':
            # L1 loss
            self.G_losses['L1'] = torch.mean(torch.abs(self.fake - self.gt)) * self.lambda_L1
            # GAN loss global
            fake_global = self.masked_fake
            pred_fake_global = self.netD_Glbal(fake_global)
            self.G_losses['G_GAN_global'] = self.criterionGAN_global(pred_fake_global, True, for_discriminator=False)
            # GAN loss comp
            fake_concat = self.fake_le
            real_concat = self.gt_le
            pred_fake, pred_real = self.discriminate(self.netD_le, fake_concat, real_concat, for_dis=False)
            self.G_losses['G_fake_le'] = self.criterionGAN_comp(pred_fake, True, for_discriminator=False) * self.cfg['lambda_comp']
            fake_concat = self.fake_re
            real_concat = self.gt_re
            pred_fake, pred_real = self.discriminate(self.netD_re, fake_concat, real_concat, for_dis=False)
            self.G_losses['G_fake_re'] = self.criterionGAN_comp(pred_fake, True, for_discriminator=False) * self.cfg['lambda_comp']
            fake_concat = self.fake_mouth
            real_concat = self.gt_mouth
            pred_fake, pred_real = self.discriminate(self.netD_mouth, fake_concat, real_concat, for_dis=False)
            self.G_losses['G_fake_mouth'] = self.criterionGAN_comp(pred_fake, True, for_discriminator=False) * self.cfg['lambda_comp']
            # arcface loss
            self.G_losses['arcface'] = self.criterionArcface(self.masked_fake, self.ref) * self.lambda_arcface

        elif train_mode == 'comp':
            # L1 loss
            self.G_losses['L1'] = torch.mean(torch.abs((self.fake - self.gt) * self.mask)) * self.lambda_L1 + \
                                    torch.mean(torch.abs((self.fake - self.gt) * (1.0 - self.mask))) * 1.0
            # GAN loss global comp
            fake_global = self.masked_fake
            pred_fake_global = self.netD_Glbal(fake_global)
            self.G_losses['G_GAN_global'] = self.criterionGAN_global(pred_fake_global, True, for_discriminator=False)
            # # GAN loss comp
            # fake_concat = self.fake_le
            # real_concat = self.gt_le
            # pred_fake, pred_real = self.discriminate(self.netD_le, fake_concat, real_concat, for_dis=False)
            # self.G_losses['G_fake_le'] = self.criterionGAN_comp(pred_fake, True, for_discriminator=False) * self.cfg['lambda_comp']
            # fake_concat = self.fake_re
            # real_concat = self.gt_re
            # pred_fake, pred_real = self.discriminate(self.netD_re, fake_concat, real_concat, for_dis=False)
            # self.G_losses['G_fake_re'] = self.criterionGAN_comp(pred_fake, True, for_discriminator=False) * self.cfg['lambda_comp']
            # fake_concat = self.fake_mouth
            # real_concat = self.gt_mouth
            # pred_fake, pred_real = self.discriminate(self.netD_mouth, fake_concat, real_concat, for_dis=False)
            # self.G_losses['G_fake_mouth'] = self.criterionGAN_comp(pred_fake, True, for_discriminator=False) * self.cfg['lambda_comp']
            # no loss
            self.G_losses['G_fake_le'] = 0.0
            self.G_losses['G_fake_re'] = 0.0
            self.G_losses['G_fake_mouth'] = 0.0
            # arcface loss
            # self.G_losses['arcface'] = self.criterionArcface(self.masked_fake, self.gt) * self.lambda_arcface
            self.G_losses['arcface'] = 0.0

        if train_mode == 'seg':
            # L1 loss
            self.G_losses['L1'] = torch.mean(torch.abs(self.fake - self.gt)) * self.lambda_L1
            # GAN loss global comp
            fake_global = self.masked_fake
            pred_fake_global = self.netD_Glbal(fake_global)
            self.G_losses['G_GAN_global'] = self.criterionGAN_global(pred_fake_global, True, for_discriminator=False)
            # no loss
            self.G_losses['G_fake_le'] = 0.0
            self.G_losses['G_fake_re'] = 0.0
            self.G_losses['G_fake_mouth'] = 0.0
            # arcface loss
            # self.G_losses['arcface'] = self.criterionArcface(self.masked_fake, self.ref) * self.lambda_arcface
            self.G_losses['arcface'] = 0.0

        # VGG loss
        if not self.cfg['no_vgg_loss']:
            self.G_losses['VGG'] = self.criterionVGG(self.fake, self.gt) * self.lambda_vgg
        
        # Parsing Losses
        # parsloss1, self.vis_fake1, self.vis_gt1, self.vis_ref1 = self.criterionParsing(self.pars1, self.pars_gt, self.pars_ref)
        parsloss2, self.vis_fake2, self.vis_gt2, self.vis_ref2 = self.criterionParsing(self.pars2, self.pars_gt, self.pars_ref)
        parsloss3, self.vis_fake3, self.vis_gt3, self.vis_ref3 = self.criterionParsing(self.pars3, self.pars_gt, self.pars_ref)
        # parsloss4, self.vis_fake4, self.vis_gt4, self.vis_ref4 = self.criterionParsing(self.pars4, self.pars_gt, self.pars_ref)

        self.G_losses['G_msParsing'] = (parsloss2 + parsloss3) * self.G_msParsing

        return self.G_losses

    def optimize_parameters(self, train_mode):
        # if train_mode == 'inpainting':
            # self.lambda_L1 = 0.0
            # self.G_msParsing = 0.2
            # self.lambda_arcface = 3.0
            # self.lambda_vgg = 10.0
            # self.set_requires_grad(self.netG.Zencoder, False)
        # elif train_mode == 'comp':
            # self.lambda_L1 = 20.0
            # self.G_msParsing = 0.5
            # self.lambda_arcface = 0.0
            # self.lambda_vgg = 5.0
            # self.set_requires_grad(self.netG.Zencoder, True)
        # elif train_mode == 'seg':
            # self.lambda_L1 = 5.0
            # self.G_msParsing = 2.5
            # self.lambda_arcface = 0.0
            # self.lambda_vgg = 1.0
            # self.set_requires_grad(self.netG.Zencoder, False)
            
        if train_mode == 'inpainting':
            self.lambda_L1 = 20.0
            self.G_msParsing = 2.5
            self.lambda_arcface = 3.0
            self.lambda_vgg = 10.0
            self.set_requires_grad(self.netG.Zencoder, False)
        elif train_mode == 'comp':
            self.lambda_L1 = 20.0
            self.G_msParsing = 2.5
            self.lambda_arcface = 3.0
            self.lambda_vgg = 10.0
            self.set_requires_grad(self.netG.Zencoder, True)
        elif train_mode == 'seg':
            self.lambda_L1 = 20.0
            self.G_msParsing = 2.5
            self.lambda_arcface = 3.0
            self.lambda_vgg = 10.0
            self.set_requires_grad(self.netG.Zencoder, False)

        self.forward(train_mode=train_mode)
        
        if train_mode == 'inpainting':
            # update global D
            self.set_requires_grad(self.netD_Glbal, True)  # enable backprop for D
            self.optimizer_D_global.zero_grad()  # set D's gradients to zero
            self.d_global_losses = self.compute_D_global_loss()
            d_global_loss = sum(self.d_global_losses.values()).mean()
            d_global_loss.backward()
            self.optimizer_D_global.step()
            # update comp D
            self.set_requires_grad(self.netD_le, True)
            self.set_requires_grad(self.netD_re, True)
            self.set_requires_grad(self.netD_mouth, True)
            self.optimizer_D_le.zero_grad()
            self.optimizer_D_re.zero_grad()
            self.optimizer_D_mouth.zero_grad()
            self.d_comp_losses = self.compute_D_comp()
            d_loss = sum(self.d_comp_losses.values()).mean()
            d_loss.backward()
            self.optimizer_D_le.step()
            self.optimizer_D_re.step()
            self.optimizer_D_mouth.step()
            
        elif train_mode == 'comp':
            # update global D
            self.set_requires_grad(self.netD_Glbal, True)  # enable backprop for D
            self.optimizer_D_global.zero_grad()  # set D's gradients to zero
            self.d_global_losses = self.compute_D_global_loss()
            d_global_loss = sum(self.d_global_losses.values()).mean()
            d_global_loss.backward()
            self.optimizer_D_global.step()
            # # update comp D
            # self.set_requires_grad(self.netD_le, True)
            # self.set_requires_grad(self.netD_re, True)
            # self.set_requires_grad(self.netD_mouth, True)
            # self.optimizer_D_le.zero_grad()
            # self.optimizer_D_re.zero_grad()
            # self.optimizer_D_mouth.zero_grad()
            # self.d_comp_losses = self.compute_D_comp()
            # d_loss = sum(self.d_comp_losses.values()).mean()
            # d_loss.backward()
            # self.optimizer_D_le.step()
            # self.optimizer_D_re.step()
            # self.optimizer_D_mouth.step()
            # no loss
            self.D_comp_losses['D_fake_le'] = 0.0
            self.D_comp_losses['D_real_le'] = 0.0
            self.D_comp_losses['D_fake_re'] = 0.0
            self.D_comp_losses['D_real_re'] = 0.0
            self.D_comp_losses['D_fake_mouth'] = 0.0
            self.D_comp_losses['D_real_mouth'] = 0.0

        elif train_mode == 'seg':
            # update global D
            self.set_requires_grad(self.netD_Glbal, True)  # enable backprop for D
            self.optimizer_D_global.zero_grad()  # set D's gradients to zero
            self.d_global_losses = self.compute_D_global_loss()
            d_global_loss = sum(self.d_global_losses.values()).mean()
            d_global_loss.backward()
            self.optimizer_D_global.step()
            # no loss
            self.D_comp_losses['D_fake_le'] = 0.0
            self.D_comp_losses['D_real_le'] = 0.0
            self.D_comp_losses['D_fake_re'] = 0.0
            self.D_comp_losses['D_real_re'] = 0.0
            self.D_comp_losses['D_fake_mouth'] = 0.0
            self.D_comp_losses['D_real_mouth'] = 0.0

            # update G
        # self.set_requires_grad(self.netD_patch, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_Glbal, False)  # D requires no gradients when optimizing G
        # self.set_requires_grad(self.netD_Glbal_comp, False)
        self.set_requires_grad(self.netD_le, False)
        self.set_requires_grad(self.netD_re, False)
        self.set_requires_grad(self.netD_mouth, False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.g_losses = self.compute_G_loss(train_mode=train_mode)
        g_loss = sum(self.g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()  # udpate G's weights

    #########################################################################################################
    ########## util func #############
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def visual_results(self):
        return {'masked_img': self.masked_img, 'gt': self.gt, 'masked_fake': self.masked_fake, 'ref': self.ref,
                'mask': self.mask, 'fake': self.fake,
                # 'vis_fake1': self.vis_fake1, 'vis_gt1': self.vis_gt1, 'vis_ref1': self.vis_ref1,
                'vis_fake2': self.vis_fake2, 'vis_gt2': self.vis_gt2, 'vis_ref2': self.vis_ref2,
                'vis_fake3': self.vis_fake3, 'vis_gt3': self.vis_gt3, 'vis_ref3': self.vis_ref3,
                # 'vis_fake4': self.vis_fake4, 'vis_gt4': self.vis_gt4, 'vis_ref4': self.vis_ref4,
                'gt_le': self.gt_le, 'gt_re': self.gt_re, 'gt_nose': self.gt_nose, 'gt_mouth': self.gt_mouth,
                'ref_le': self.ref_le, 'ref_re': self.ref_re, 'ref_nose': self.ref_nose, 'ref_mouth': self.ref_mouth,
                'fake_le': self.fake_le, 'fake_re': self.fake_re, 'fake_nose': self.fake_nose, 'fake_mouth': self.fake_mouth,
                'style_map_select_2': self.style_map_select_2,
                'style_map_select_3': self.style_map_select_3,
                'style_map_mask_2': self.style_map_mask_2,
                'style_map_mask_3': self.style_map_mask_3,
                }, self.name

    def print_losses(self):
        print('G Losses')
        for v, k in self.G_losses.items():
            print(v, ': ', k)

        print('D global Losses')
        for v, k in self.D_global_losses.items():
            print(v, ': ', k)

        print('D patch Losses')
        for v, k in self.D_patch_losses.items():
            print(v, ': ', k)

        print('D comp Losses')
        for v, k in self.D_comp_losses.items():
            print(v, ': ', k)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def compute_arcface_loss(self):
        # print('ref_gt')
        ref_gt = self.criterionArcface(self.ref, self.gt)
        # print('fake_gt')
        fake_gt = self.criterionArcface(self.masked_fake, self.gt)
        # print('fake_ref')
        fake_ref = self.criterionArcface(self.masked_fake, self.ref)
        return ref_gt, fake_gt, fake_ref

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def save_nets(self, epoch, cfg, suffix=''):
        save_file = {}
        save_file['epoch'] = epoch
        for name in self.model_names:
            net = getattr(self, name)
            save_file[name] = net.cpu().state_dict()
            if torch.cuda.is_available():
                net.cuda()
        for name in self.opt_names:
            opt = getattr(self, name)
            save_file[name] = opt.state_dict()
        save_filename = '%03d_ckpt_%s.pth' % (epoch, suffix)
        save_path = os.path.join(cfg['checkpoints_dir'], save_filename)
        torch.save(save_file, save_path)

    def save_latest_nets(self, epoch, cfg):
        save_file = {}
        save_file['epoch'] = epoch
        for name in self.model_names:
            net = getattr(self, name)
            save_file[name] = net.cpu().state_dict()
            if torch.cuda.is_available():
                net.cuda()
        for name in self.opt_names:
            opt = getattr(self, name)
            save_file[name] = opt.state_dict()
        save_filename = 'latest_ckpt.pth'
        save_path = os.path.join(cfg['checkpoints_dir'], save_filename)
        torch.save(save_file, save_path)

    def print_networks(self):
        """Print the total number of parameters in the network and network architecture"""
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                # print network architecture
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_train(self):
        for name in self.model_names:
            net = getattr(self, name)
            net.train()
        self.arcface.eval()

    def set_eval(self):
        for name in self.model_names:
            net = getattr(self, name)
            net.eval()
        self.arcface.eval()

    def comp_crop(self):
        # gt
        gt_le = []
        for n in range(self.gt.size()[0]):
            curr_bbox = self.gt_le_bbox[n]
            # print(curr_bbox)
            curr_le = self.gt[n:n+1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_le = F.interpolate(curr_le, self.cfg['EYE_shape'], mode='bilinear')
            gt_le.append(curr_le)
        self.gt_le = torch.cat(gt_le, dim=0)

        gt_re = []
        for n in range(self.gt.size()[0]):
            curr_bbox = self.gt_re_bbox[n]
            # print(curr_bbox)
            curr_re = self.gt[n:n + 1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_re = F.interpolate(curr_re, self.cfg['EYE_shape'], mode='bilinear')
            gt_re.append(curr_re)
        self.gt_re = torch.cat(gt_re, dim=0)

        gt_nose = []
        for n in range(self.gt.size()[0]):
            curr_bbox = self.gt_nose_bbox[n]
            # print(curr_bbox)
            curr_nose = self.gt[n:n + 1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_nose = F.interpolate(curr_nose, self.cfg['NOSE_shape'], mode='bilinear')
            gt_nose.append(curr_nose)
        self.gt_nose = torch.cat(gt_nose, dim=0)

        gt_mouth = []
        for n in range(self.gt.size()[0]):
            curr_bbox = self.gt_mouth_bbox[n]
            # print(curr_bbox)
            curr_mouth = self.gt[n:n + 1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_mouth = F.interpolate(curr_mouth, self.cfg['MOUTH_shape'], mode='bilinear')
            gt_mouth.append(curr_mouth)
        self.gt_mouth = torch.cat(gt_mouth, dim=0)

        # ref
        ref_le = []
        for n in range(self.ref.size()[0]):
            curr_bbox = self.ref_le_bbox[n]
            # print(curr_bbox)
            curr_le = self.ref[n:n+1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_le = F.interpolate(curr_le, self.cfg['EYE_shape'], mode='bilinear')
            ref_le.append(curr_le)
        self.ref_le = torch.cat(ref_le, dim=0)

        ref_re = []
        for n in range(self.ref.size()[0]):
            curr_bbox = self.ref_re_bbox[n]
            # print(curr_bbox)
            curr_re = self.ref[n:n + 1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_re = F.interpolate(curr_re, self.cfg['EYE_shape'], mode='bilinear')
            ref_re.append(curr_re)
        self.ref_re = torch.cat(ref_re, dim=0)

        ref_nose = []
        for n in range(self.ref.size()[0]):
            curr_bbox = self.ref_nose_bbox[n]
            # print(curr_bbox)
            curr_nose = self.ref[n:n + 1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_nose = F.interpolate(curr_nose, self.cfg['NOSE_shape'], mode='bilinear')
            ref_nose.append(curr_nose)
        self.ref_nose = torch.cat(ref_nose, dim=0)

        ref_mouth = []
        for n in range(self.ref.size()[0]):
            curr_bbox = self.ref_mouth_bbox[n]
            # print(curr_bbox)
            curr_mouth = self.ref[n:n + 1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_mouth = F.interpolate(curr_mouth, self.cfg['MOUTH_shape'], mode='bilinear')
            ref_mouth.append(curr_mouth)
        self.ref_mouth = torch.cat(ref_mouth, dim=0)

        # fake
        fake_le = []
        for n in range(self.fake.size()[0]):
            curr_bbox = self.gt_le_bbox[n]
            curr_le = self.fake[n:n+1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_le = F.interpolate(curr_le, self.cfg['EYE_shape'], mode='bilinear')
            fake_le.append(curr_le)
        self.fake_le = torch.cat(fake_le, dim=0)

        fake_re = []
        for n in range(self.fake.size()[0]):
            curr_bbox = self.gt_re_bbox[n]
            curr_re = self.fake[n:n + 1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_re = F.interpolate(curr_re, self.cfg['EYE_shape'], mode='bilinear')
            fake_re.append(curr_re)
        self.fake_re = torch.cat(fake_re, dim=0)

        fake_nose = []
        for n in range(self.fake.size()[0]):
            curr_bbox = self.gt_nose_bbox[n]
            curr_nose = self.fake[n:n + 1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_nose = F.interpolate(curr_nose, self.cfg['NOSE_shape'], mode='bilinear')
            fake_nose.append(curr_nose)
        self.fake_nose = torch.cat(fake_nose, dim=0)

        fake_mouth = []
        for n in range(self.fake.size()[0]):
            curr_bbox = self.gt_mouth_bbox[n]
            curr_mouth = self.fake[n:n + 1, :, curr_bbox[0]:curr_bbox[1], curr_bbox[2]:curr_bbox[3]]
            curr_mouth = F.interpolate(curr_mouth, self.cfg['MOUTH_shape'], mode='bilinear')
            fake_mouth.append(curr_mouth)
        self.fake_mouth = torch.cat(fake_mouth, dim=0)

    def resume(self, checkpoint_dir, ckpt_filename=None):
        if not ckpt_filename:
            ckpt_filename = 'latest_ckpt.pth'
        ckpt = torch.load(os.path.join(checkpoint_dir, ckpt_filename))
        cur_epoch = ckpt['epoch']
        for name in self.model_names:
            net = getattr(self, name)
            net.load_state_dict(ckpt[name])
            print('load model %s of epoch %d' % (name, cur_epoch))
        for name in self.opt_names:
            opt = getattr(self, name)
            opt.load_state_dict(ckpt[name])
            print('load opt %s of epoch %d' % (name, cur_epoch))
        return cur_epoch

    def vis_segmap(self, segres):
        part_colors = [[0, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]

        parsing = segres.cpu().detach().numpy().argmax(0)
        vis_parsing_anno = parsing.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, (256,256), interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = 19

        for pi in range(0, num_of_class):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

        return vis_parsing_anno_color