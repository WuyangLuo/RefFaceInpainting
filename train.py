from utils import *
import argparse
import numpy as np
import datetime
from trainer import Trainer
from dataset import ID_Dataset
import torch
from torch.utils.data import DataLoader
import os
import shutil
import cv2
from metrics.fid_score import calculate_fid_given_paths

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

torch.multiprocessing.set_sharing_strategy('file_system')  # RuntimeError: received 0 items of ancdata

import sys
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--resume_dir', type=str, default='outputs/2022-01-25_18-29-24/', help="outputs path")
opts = parser.parse_args()

print_options(opts)

# Load experiment setting
cfg = get_config(opts.config)

# set hyperparameter
batch_size = cfg['batch_size']
shuffle = cfg['shuffle']
max_epoch = cfg['max_epoch']

trainer = Trainer(cfg)
trainer.cuda()

# print model information
trainer.print_networks()

# Setup dataset
dataset_root = cfg['dataset_dir']
train_dataset = ID_Dataset(cfg, dataset_root, split='train')
train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=cfg['worker'], pin_memory=True)
test_dataset = ID_Dataset(cfg, dataset_root, split='test')
test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test_batch_size'], shuffle=False, num_workers=cfg['worker'])
print('train dataset containing ', len(train_loader), 'images')
print('test dataset containing ', len(test_loader), 'images')

# Setup logger and output folders
if opts.resume:
    checkpoint_directory = opts.resume_dir + 'checkpoints/'
    image_directory = opts.resume_dir + 'images/'
    result_directory = opts.resume_dir + 'results/'
    cur_epoch = trainer.resume(checkpoint_directory, ckpt_filename=None) + 1
    shutil.copy(opts.config, os.path.join(opts.resume_dir, 'config_resume.yaml'))  # copy config file to output folder
else:
    cur_epoch = 0
    output_directory = os.path.join(opts.output_path + "/outputs", datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    checkpoint_directory, image_directory, result_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

cfg['checkpoints_dir'] = checkpoint_directory

best_fid = float("inf")

# Start training
print('training start at %d th epoch'%(cur_epoch))

for epoch in range(cur_epoch, cfg['max_epoch']):
    for i, data in enumerate(train_loader):  # inner loop within one epoch
        trainer.set_train()
        trainer.set_input(data)  # unpack data from dataset and apply preprocessing
        
        visual_iter_num = 4
        if i % visual_iter_num == 0 or i % visual_iter_num == 2:
            trainer.optimize_parameters(train_mode='inpainting')
        elif i % visual_iter_num == 1:
            trainer.optimize_parameters(train_mode='comp')
        elif i % visual_iter_num == 3:
            trainer.optimize_parameters(train_mode='seg')

        if i % cfg['visual_img_freq'] < visual_iter_num:  # display images on visdom and save images to a HTML file
            results, img_name = trainer.visual_results()
    
            cur_img_dir = os.path.join(image_directory, 'epoch-%d_iter-%d_%s'%(epoch, i, img_name[0]))
            if not os.path.exists(cur_img_dir):
                os.makedirs(cur_img_dir)
    
            is_fg = ['mask', 'lab']
            for name, img in results.items():
                no_fg = True
                if name in is_fg:
                    no_fg = False
                save_name = 'epoch-%d_iter-%d_%s_'%(epoch, i, name)+'.jpg'
                if name == 'lab':
                    lab = lab2im(img)
                    cv2.imwrite(os.path.join(cur_img_dir, save_name), lab)
                    label_dir = os.path.join(cur_img_dir, 'label')
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir)
                    set = np.unique(lab)
                    for l in set:
                        cur_lab = np.array(np.equal(lab, l).astype(np.uint8)) * 255
                        cv2.imwrite(os.path.join(label_dir, str(l)+'.jpg'), cur_lab)
                elif name == 'att':
                    att_dir = os.path.join(cur_img_dir, 'att')
                    if not os.path.exists(att_dir):
                        os.makedirs(att_dir)
                    for i_att, att in enumerate(img):
                        hm = tensor2hm(att)
                        cv2.imwrite(os.path.join(att_dir, 'att_'+str(i_att)+'.jpg'), cv2.applyColorMap(hm, cv2.COLORMAP_JET))
                elif 'style' in name:
                    cv2.imwrite(os.path.join(cur_img_dir, save_name), tensor2im(img[:, :3], no_fg=no_fg))
                elif name == 'masks':
                    masks_dir = os.path.join(cur_img_dir, 'masks')
                    if not os.path.exists(masks_dir):
                        os.makedirs(masks_dir)
                    for idx, m in enumerate(img):
                        hm = tensor2hm(m)
                        if idx % 2 == 0:
                            cv2.imwrite(os.path.join(masks_dir, 'att_'+str(str(int(idx/2))+'_1')+'.png'), cv2.applyColorMap(hm, cv2.COLORMAP_JET))
                        elif idx % 2 == 1:
                            cv2.imwrite(os.path.join(masks_dir, 'att_'+str(str(int(idx/2))+'_2')+'.png'), cv2.applyColorMap(hm, cv2.COLORMAP_JET))
                else:
                    cv2.imwrite(os.path.join(cur_img_dir, save_name), tensor2im(img, no_fg=no_fg))
    
        if i % cfg['print_loss_freq'] < visual_iter_num:  # print training losses and save logging information to the disk
            print('print losses at the {} epoch {} iter'.format(epoch, i))
            trainer.print_losses()
    
    if (epoch+1) % cfg['save_epoch_freq'] == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d' % (epoch))
        trainer.save_nets(epoch, cfg)
    
    print('saving the model at {} epoch'.format(epoch))
    trainer.save_latest_nets(epoch, cfg)


    # test
    if epoch % cfg['test_freq'] == 0:
        with torch.no_grad():
            print('testing at %d epoch' % (epoch))
            trainer.set_eval()
            cur_save_dir = os.path.join(result_directory, str(epoch))
            if not os.path.exists(cur_save_dir):
                os.makedirs(cur_save_dir)
            for i, data in enumerate(test_loader):
                trainer.set_input(data)
                trainer.forward_test()
                results, img_name = trainer.visual_results()
                cv2.imwrite(os.path.join(cur_save_dir, img_name[0]+'.jpg'), tensor2im(results['masked_fake']))

            path_gt = os.path.join(cfg['dataset_dir'], 'test_gt')
            path_test = cur_save_dir
            print('path_gt: ', path_gt)
            print('path_test: ', path_test)

            # compute FID score
            path = [path_gt, path_test]
            fid_value = calculate_fid_given_paths(path, cfg['test_batch_size'])
            print('========FID==========: ', fid_value)

            if fid_value < best_fid:
                print('saving the current best model at the end of epoch %d' % (epoch))
                trainer.save_nets(epoch, cfg, suffix='_best_FID_'+str(fid_value))
                best_fid = fid_value

