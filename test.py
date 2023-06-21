import numpy as np
import datetime
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import shutil
import cv2
from utils import *
from trainer import Trainer


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


gt_img_path = 'test_gt_subtest/'     # path to test_gt_subtest
ref_path = 'test_ref/'               # path to test_ref
lab_path = 'ref_labels/'             # path to ref_labels

mask_path = 'test_masks/'            # path to test_masks

save_path = 'results_multi_ref_subset/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
img_list = os.listdir(gt_img_path)[:50]
img_list.sort()

# =======================================================================================================
# =======================================================================================================

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(img)
    img = img.cuda().unsqueeze(0)
    return img

def get_transform_lab(lab):
    lab = transforms.ToTensor()(lab)
    lab = lab.cuda().unsqueeze(0)
    return lab

def eval_model(trainer):
    for idx, img_name in enumerate(img_list):
        for idx_ref, ref_name in enumerate(img_list):
            # print(idx_ref, ref_name)
            img = cv2.imread(os.path.join(gt_img_path, img_name))
            img = preprocess(img)
            ref = cv2.imread(os.path.join(ref_path, ref_name))
            ref = preprocess(ref)
            
            lab_ref = cv2.imread(lab_path + ref_name[:-4]+'.png', 0)
            lab_ref = get_transform_lab(lab_ref)
            lab_ref = lab_ref * 255.0
            
            mask = cv2.imread(os.path.join(mask_path, '%03d'%(idx)+'.png'), 0)
            mask = mask.reshape((1,) + mask.shape).astype(np.float32) / 255
            mask = torch.from_numpy(mask).unsqueeze(0).cuda()

            masked_img = img * (1. - mask)

            masked_fake = trainer.test(img, masked_img, ref, mask, lab_ref)
            
            size = 256
            n = 4
            m = 3
            img_concat = np.ones((size, size*n + m*(n-1), 3)) * 255
            
            print(masked_img.size())
            masked_img = tensor2im(masked_img, no_fg=True)
            ref = tensor2im(ref, no_fg=True)
            gt = tensor2im(img, no_fg=True)
            masked_fake = tensor2im(masked_fake, no_fg=True)
            
            print('{}_{}.jpg'.format(idx, idx_ref))
            curr_save_dir = os.path.join(save_path, str(idx))
            if not os.path.exists(curr_save_dir):
                os.makedirs(curr_save_dir)
            cv2.imwrite(os.path.join(curr_save_dir, 'gt_{}.jpg'.format(idx)), gt)
            cv2.imwrite(os.path.join(curr_save_dir, 'ref_{}.jpg'.format(idx_ref)), ref)
            cv2.imwrite(os.path.join(curr_save_dir, 'masked_img.jpg'), masked_img)
            cv2.imwrite(os.path.join(curr_save_dir, '{}.jpg'.format(idx_ref)), masked_fake)


cfg = get_config('configs/config.yaml')
model_path = 'model.pth'  # trained model
trainer = Trainer(cfg)
trainer.cuda()
trainer.netG.load_state_dict(torch.load(model_path)['netG'])
trainer.set_eval()
eval_model(trainer)





