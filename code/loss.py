import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.blocks import VGG19
from networks.vgg_face import VGG_FACE
import numpy as np
import cv2


# # Defines the GAN loss which uses either LSGAN or the regular GAN.
# # When LSGAN is used, it is basically same as MSELoss,
# # but it abstracts away the need to create the target label tensor
# # that has the same size as the input
class GANLoss_MultiD(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss_MultiD, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'vanilla':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'vanilla':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))      # real:让input大于1即可
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))     # fake: 让input小于-1即可
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'hinge']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, for_discriminator=True):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(prediction - 1, self.get_target_tensor(prediction, False))      # real:让input大于1即可
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-prediction - 1, self.get_target_tensor(prediction, False))     # fake: 让input小于-1即可
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(prediction)
            return loss
        return loss


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def classification_loss(self, logit, target):
    """Compute binary or softmax cross entropy loss."""
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


def FeatMatchLoss(x, y):
    criterion = nn.L1Loss()
    weights = [1.0 / 2, 1.0 / 4, 1.0 / 8, 1.0 / 16]
    loss = 0
    for i in range(len(x)):
        loss += weights[i] * criterion(x[i], y[i].detach())
    return loss


class VGGfaceLoss(nn.Module):
    def __init__(self):
        super(VGGfaceLoss, self).__init__()
        self.vgg_face = VGG_FACE().cuda()
        self.criterion = nn.MSELoss()

        self.mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1).cuda()

    def preprocessing(self, x):
        x = x * 255 - self.mean
        x = F.interpolate(x, size=(224,224), mode='bilinear')
        return x

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg_face(x), self.vgg_face(y)
        loss = 0
        loss = self.criterion(self.preprocessing(x_vgg), self.preprocessing(y_vgg.detach()))
        return loss


class Arcface_Loss(nn.Module):
    def __init__(self, net):
        super(Arcface_Loss, self).__init__()
        self.net = net
        self.l1 = nn.L1Loss()

    def forward(self, fake, gt):
        self.net.eval()
        id_fake = self.net(F.interpolate(fake, size=112, mode='bilinear'))
        id_gt = self.net(F.interpolate(gt, size=112, mode='bilinear'))

        id_fake = F.normalize(id_fake)
        id_gt = F.normalize(id_gt)

        # print(self.net.state_dict()['conv1.weight'][0])
        
        # print(self.net.training)
        
        inner_product = (torch.bmm(id_fake.unsqueeze(1), id_gt.unsqueeze(2)).squeeze())
        # print(inner_product)
        return self.l1(torch.ones_like(inner_product), inner_product)


class Parsing_Loss(nn.Module):
    def __init__(self, num_of_class=19):
        super(Parsing_Loss, self).__init__()
        self.num_of_class = num_of_class
        self.CEloss = nn.CrossEntropyLoss(ignore_index=-100)

    def vis_map(self, segres, is_gt=False):
        if self.num_of_class == 19:
            part_colors = [[0, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]
        if not is_gt:
            parsing = segres.cpu().detach().numpy().argmax(0)
            vis_parsing_anno = parsing.copy().astype(np.uint8)
            vis_parsing_anno = cv2.resize(vis_parsing_anno, (256, 256), interpolation=cv2.INTER_NEAREST)
            vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

            for pi in range(0, self.num_of_class + 1):
                index = np.where(vis_parsing_anno == pi)
                vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

            vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        else:
            parsing = segres.cpu().detach().numpy()
            vis_parsing_anno = parsing.copy().astype(np.uint8)
            vis_parsing_anno = cv2.resize(vis_parsing_anno, (256, 256), interpolation=cv2.INTER_NEAREST)
            vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

            for pi in range(0, self.num_of_class + 1):
                index = np.where(vis_parsing_anno == pi)
                vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

            vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)


        return vis_parsing_anno_color

    def forward(self, logit, gt_lab, ref_lab):
        gt_lab = gt_lab.unsqueeze(1)
        gt_lab = gt_lab.float()

        gt_lab = F.interpolate(gt_lab, size=logit.size()[-1], mode='nearest').squeeze(1)
        gt_lab = gt_lab.long()
        
        loss = self.CEloss(logit, gt_lab)

        vis_fake = self.vis_map(logit[0])
        vis_gt = self.vis_map(gt_lab[0], is_gt=True)

        ref_lab = ref_lab.unsqueeze(1)
        ref_lab = ref_lab.float()
        ref_lab = F.interpolate(ref_lab, size=logit.size()[-1], mode='nearest').squeeze(1)
        ref_lab = ref_lab.long()
        vis_ref = self.vis_map(ref_lab[0], is_gt=True)

        return loss, vis_fake, vis_gt, vis_ref



class Rec_Loss(nn.Module):
    def __init__(self):
        super(Rec_Loss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, fake, gt):

        gt = F.interpolate(gt, size=fake.size()[2:], mode='bilinear')
        loss = self.l1(fake, gt)

        return loss