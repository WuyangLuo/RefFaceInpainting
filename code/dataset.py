import os
import cv2
import math
import numpy as np
from torch.utils.data import Dataset
import os.path
import random
import torchvision.transforms as transforms
import torch
from utils import make_dataset
import pickle
from PIL import Image, ImageDraw


class ID_Dataset(Dataset):
    def __init__(self, cfg, dataset_root, split='train', mask_type='ff_mask'):
        self.split = split
        self.cfg = cfg
        self.dataset_root = dataset_root
        self.mask_type = mask_type

        list_path = os.path.join(self.dataset_root, self.split+'_list.pkl')
        f = open(list_path, "rb")
        self.name_list = pickle.load(f)
        f.close()        
        
        self.img_path = os.path.join(self.dataset_root, 'images/')
        self.anno_path = os.path.join(self.dataset_root, 'annotations/')
        self.id_path = os.path.join(self.dataset_root, 'id_list/')
        self.lab_path = os.path.join(self.dataset_root, 'labels/')

        if self.mask_type == 'ff_mask':
            self.mask_path = os.path.join(self.dataset_root, 'ff_mask/')
            mask_list = os.listdir(self.mask_path)
            mask_list.sort()
            self.mask_list = mask_list
        else:
            pass


    def __getitem__(self, index):
        name = self.name_list[index]

        f_id = open(os.path.join(self.id_path, name+'.pkl'), "rb")
        id_list = pickle.load(f_id)
        f_id.close()
        
        if self.split == 'train':
            corr_name, ref_name = random.sample(id_list, 2)
        elif self.split == 'test':
            corr_name, ref_name = id_list[0], id_list[1]

        img_corr = cv2.imread(self.img_path + corr_name+'.jpg')
        img_ref = cv2.imread(self.img_path + ref_name+'.jpg')

        img_corr = cv2.cvtColor(img_corr, cv2.COLOR_BGR2RGB)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

        f_corr = open(os.path.join(self.anno_path, corr_name+'.pkl'), "rb")
        anno_corr = pickle.load(f_corr)
        f_corr.close()

        f_ref = open(os.path.join(self.anno_path, ref_name+'.pkl'), "rb")
        anno_ref = pickle.load(f_ref)
        f_ref.close()

        img_corr = get_transform(img_corr)
        img_ref = get_transform(img_ref)

        mask, bbox = self.load_mask(self.cfg['crop_size'], anno_corr["points"]["n_c"])

        if self.mask_type == 'ff_mask':
            if self.split == 'train':
                mask = brush_stroke_mask()
                mask = mask.reshape((1,) + mask.shape).astype(np.float32)
                mask_seg = brush_stroke_mask_seg()
                mask_seg = mask_seg.reshape((1,) + mask_seg.shape).astype(np.float32)
            elif self.split == 'test':
                mask = cv2.imread(os.path.join(self.mask_path, self.mask_list[index]), 0)
                mask = mask.reshape((1,) + mask.shape).astype(np.float32)
                mask_seg = mask
        else:
            pass
        
        mask = torch.from_numpy(mask)
        masked_img = img_corr * (1. - mask)

        lab_gt = cv2.imread(self.lab_path + corr_name+'.png', 0)
        pars_gt = lab_gt
        lab_gt = get_transform(lab_gt, normalize=False)
        lab_gt = lab_gt * 255.0
        
        lab_ref = cv2.imread(self.lab_path + ref_name+'.png', 0)
        pars_ref = lab_ref
        lab_ref = get_transform(lab_ref, normalize=False)
        lab_ref = lab_ref * 255.0
        
        # lab_gt = lab_gt + 1
        # lab_ref = lab_ref + 1
        
        data = {}
        data['img_corr'] = img_corr
        data['img_ref'] = img_ref
        data['masked_img'] = masked_img
        data['mask'] = mask
        data['mask_seg'] = mask_seg
        data['bbox'] = bbox
        data['name'] = name
        data['lab_gt'] = lab_gt
        data['lab_ref'] = lab_ref
        data['pars_gt'] = pars_gt
        data['pars_ref'] = pars_ref

        data['gt_le'] = np.array([anno_corr["bbox"]["le"]["lu"][1], anno_corr["bbox"]["le"]["rb"][1],
                                  anno_corr["bbox"]["le"]["lu"][0], anno_corr["bbox"]["le"]["rb"][0]])
        data['gt_re'] = np.array([anno_corr["bbox"]["re"]["lu"][1], anno_corr["bbox"]["re"]["rb"][1],
                                  anno_corr["bbox"]["re"]["lu"][0], anno_corr["bbox"]["re"]["rb"][0]])
        data['gt_nose'] = np.array([anno_corr["bbox"]["nose"]["lu"][1], anno_corr["bbox"]["nose"]["rb"][1],
                                    anno_corr["bbox"]["nose"]["lu"][0], anno_corr["bbox"]["nose"]["rb"][0]])
        data['gt_mouth'] = np.array([anno_corr["bbox"]["mouth"]["lu"][1], anno_corr["bbox"]["mouth"]["rb"][1],
                                     anno_corr["bbox"]["mouth"]["lu"][0], anno_corr["bbox"]["mouth"]["rb"][0]])

        data['ref_le'] = np.array([anno_ref["bbox"]["le"]["lu"][1], anno_ref["bbox"]["le"]["rb"][1],
                                  anno_ref["bbox"]["le"]["lu"][0], anno_ref["bbox"]["le"]["rb"][0]])
        data['ref_re'] = np.array([anno_ref["bbox"]["re"]["lu"][1], anno_ref["bbox"]["re"]["rb"][1],
                                  anno_ref["bbox"]["re"]["lu"][0], anno_ref["bbox"]["re"]["rb"][0]])
        data['ref_nose'] = np.array([anno_ref["bbox"]["nose"]["lu"][1], anno_ref["bbox"]["nose"]["rb"][1],
                                    anno_ref["bbox"]["nose"]["lu"][0], anno_ref["bbox"]["nose"]["rb"][0]])
        data['ref_mouth'] = np.array([anno_ref["bbox"]["mouth"]["lu"][1], anno_ref["bbox"]["mouth"]["rb"][1],
                                      anno_ref["bbox"]["mouth"]["lu"][0], anno_ref["bbox"]["mouth"]["rb"][0]])
        
        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.name_list)


    def load_mask(self, crop_size, c):
        height, width = crop_size, crop_size
        mh, mw = self.cfg['mask_shape']
        mask = np.zeros((crop_size, crop_size), np.float32)

        hl = max(0, c[1] - int(mh/2))
        hr = min(c[1] + int(mh/2), height)
        wl = max(0, c[0] - int(mw/2))
        wr = min(c[0] + int(mw/2), width)
        mask[hl:hr, wl:wr] = 1.  # mask区域设置为1，其他为0

        mask = mask.reshape((1,) + mask.shape).astype(np.float32)
        

        return torch.from_numpy(mask), np.array([hl,hr,wl,wr])


def get_params(cfg, size, split):
    h, w = size[:2]
    new_h = h
    new_w = w
    if split == 'train':
        if cfg['crop']:
            if new_w > 256:
                x = random.randint(0, np.maximum(0, new_w - cfg['crop_size']))
                y = 0
            else:
                x = 0
                y = random.randint(0, np.maximum(0, new_h - cfg['crop_size']))
        if cfg['flip']:
            flip = random.random() > 0.5
    elif split == 'test':
        x = 0
        y = 0
        flip = False
    return {'crop_pos': (x, y), 'flip': flip}


def preprocess(cfg, img, params):
    if cfg['crop']:
        img = __crop(img, params['crop_pos'], cfg['crop_size'])
    if cfg['flip']:
        img = __flip(img, params['flip'])
    return img


def __crop(img, pos, crop_size):
    oh, ow = img.shape[:2]

    y1, x1 = pos
    tw = th = crop_size
    if (ow >= tw or oh >= th):
        if img.ndim == 3:
            return img[x1:x1 + crop_size, y1:y1 + crop_size, :]
        else:
            return img[x1:x1 + crop_size, y1:y1 + crop_size]
    return img


def __flip(img, flip):
    if flip:
        return cv2.flip(img, 1)
    return img


def get_transform(img, normalize=True):
    transform_list = []

    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)(img)


def brush_stroke_mask(H=256, W=256):
    min_num_vertex = 14
    max_num_vertex = 16
    mean_angle = 0
    angle_range = 2 * math.pi
    min_width = 110
    max_width = 160

    average_radius = math.sqrt(H * H + W * W) / 8
    mask = Image.new('L', (W, H), 0)

    num_vertex = random.randint(min_num_vertex, max_num_vertex)
    angle_min = mean_angle - random.uniform(0, angle_range)
    angle_max = mean_angle + random.uniform(0, angle_range)
    angles = []
    vertex = []
    for i in range(num_vertex):
        if i % 2 == 0:
            angles.append(2 * math.pi - random.uniform(angle_min, angle_max))
        else:
            angles.append(random.uniform(angle_min, angle_max))

    h, w = mask.size
    vertex.append((int(random.randint(117, 139)), int(random.randint(117, 139))))
    for i in range(num_vertex):
        r = np.clip(
            np.random.normal(loc=average_radius, scale=average_radius // 2),
            0, 2 * average_radius)
        new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 110, w - 110)
        new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 120, h - 90)
        vertex.append((int(new_x), int(new_y)))

    draw = ImageDraw.Draw(mask)
    width = int(random.uniform(min_width, max_width))
    draw.line(vertex, fill=1, width=width)
    for v in vertex:
        draw.ellipse((v[0] - width // 2,
                      v[1] - width // 2,
                      v[0] + width // 2,
                      v[1] + width // 2),
                     fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)

    mask = np.asarray(mask, np.float32)

    return mask


def brush_stroke_mask_seg(H=256, W=256):
    min_num_vertex = 15
    max_num_vertex = 20
    mean_angle = 0
    angle_range = 2 * math.pi
    min_width = 20
    max_width = 40

    average_radius = math.sqrt(H * H + W * W) / 8
    mask = Image.new('L', (W, H), 0)

    num_vertex = random.randint(min_num_vertex, max_num_vertex)
    angle_min = mean_angle - random.uniform(0, angle_range)
    angle_max = mean_angle + random.uniform(0, angle_range)
    angles = []
    vertex = []
    for i in range(num_vertex):
        if i % 2 == 0:
            angles.append(2 * math.pi - random.uniform(angle_min, angle_max))
        else:
            angles.append(random.uniform(angle_min, angle_max))

    h, w = mask.size
    m = 20
    vertex.append((int(random.randint(m, w-m)), int(random.randint(m, h-m))))
    for i in range(num_vertex):
        r = np.clip(
            np.random.normal(loc=average_radius, scale=average_radius // 2),
            0, 2 * average_radius)
        new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), m, w-m)
        new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), m, h-m)
        vertex.append((int(new_x), int(new_y)))

    draw = ImageDraw.Draw(mask)
    width = int(random.uniform(min_width, max_width))
    draw.line(vertex, fill=1, width=width)
    for v in vertex:
        draw.ellipse((v[0] - width // 2,
                      v[1] - width // 2,
                      v[0] + width // 2,
                      v[1] + width // 2),
                     fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)

    mask = np.asarray(mask, np.float32)

    return mask