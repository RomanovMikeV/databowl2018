import sys
sys.path.append('../src/')
import json
import os
import random
import skimage.transform
import skimage.io
import numpy
import utils
from copy import deepcopy

import torch

def correct_coordinate(x, left_pad, right_pad):
    return (x - left_pad) / (1.0 - left_pad - right_pad)

def pad_image(img, masks, pads):
    
    left_pad = pads[0]
    top_pad = pads[1]
    right_pad = pads[2]
    bottom_pad = pads[3]
    
    actual_masks = []
    for mask in masks:
        mask[0][0] = correct_coordinate(mask[0][0], left_pad, right_pad)
        mask[0][1] = correct_coordinate(mask[0][1], top_pad, bottom_pad)
        mask[0][2] = correct_coordinate(mask[0][2], left_pad, right_pad)
        mask[0][3] = correct_coordinate(mask[0][3], top_pad, bottom_pad)
            
        if not (mask[0][0] >= 1.0 or mask[0][1] >= 1.0 or
                mask[0][2] <= 0.0 or mask[0][3] <= 0.0):
            actual_masks.append(mask)
    
    left_pad = left_pad * (img.shape[0] - 1)
    right_pad = (1.0 - right_pad) * (img.shape[0] - 1)
    top_pad = top_pad * (img.shape[1] - 1)
    bottom_pad = (1.0 - bottom_pad) * (img.shape[1] - 1)
        
    img = img[round(left_pad):round(right_pad) + 1, 
              round(top_pad):round(bottom_pad) + 1, :]
    
    return img, actual_masks
#     img = skimage.transform.resize(img, img_shape, mode='reflect')
        
def correct_masks(masks):
    new_masks = []
    for index in range(len(masks)):
        mask = masks[index]
        if not (mask[0][0] >= 0.0 and
                mask[0][1] >= 0.0 and
                mask[0][2] <= 1.0 and
                mask[0][3] <= 1.0):
            new_mask = [deepcopy(mask[0]), deepcopy(mask[1])]
        
            if mask[0][0] < 0.0:
                new_mask[0][0] = 0.0
        
            if mask[0][1] < 0.0:
                new_mask[0][1] = 0.0
        
            if mask[0][2] > 1.0:
                new_mask[0][2] = 1.0
        
            if mask[0][3] > 1.0:
                new_mask[0][3] = 1.0
                
            old_mask_width = mask[0][2] - mask[0][0]
            old_mask_height = mask[0][3] - mask[0][1]
#             new_mask[1]
            
            new_x_start = round(
                (new_mask[0][0] - mask[0][0]) / 
                old_mask_width * (mask[1].shape[0] - 1))
            new_y_start = round(
                (new_mask[0][1] - mask[0][1]) / 
                old_mask_height * (mask[1].shape[1]- 1))
            new_x_end = round(
                (new_mask[0][2] - mask[0][0]) / 
                old_mask_width * (mask[1].shape[0] - 1))
            new_y_end = round(
                (new_mask[0][3] - mask[0][1]) / 
                old_mask_height * (mask[1].shape[1] - 1))
            
            new_mask[1] = mask[1][new_x_start:new_x_end + 1, new_y_start:new_y_end + 1]
            
            
            
            if new_mask[1].shape[0] > 1 and new_mask[1].shape[1] > 1:
                new_masks.append(new_mask)
        else:
            new_masks.append(mask)
        
    return new_masks

def masks_less(mask1, mask2):
    if mask1[2][0] < mask2[2][0]:
        return True
    elif mask1[2][0] == mask2[2][0]:
        if mask1[2][1] < mask2[2][1]:
            return True
    return False


def masks_to_targets(masks, 
                     anchors=[[1.0, 1.0]], 
                     mask_shape=[9, 9], 
                     img_size=[112, 112],
                     spixel_size=[4.0, 4.0]):
    
    for index in range(len(masks)):
        x_center = (masks[index][0] + masks[index][2]) / 2.0
        y_center = (masks[index][1] + masks[index][3]) / 2.0
    
        width = (max(masks[index][2], masks[index][0]) - x_center)
        height = (max(masks[index][3], masks[index][1]) - y_center)
        
        masks[index].append([x_center, y_cetner, width, height])
    
    
    sorted(masks, )
    targets = numpy.zeros(img_size[0], img_size[1], (5 + mask_shape[1]) * len(anchors))
    
    for x_index in range(int(img_size[0] / spixel_size[0])):
        for y_index in range(int(img_size[1] / spixel_size[1])):
            
            spixel_x_center = x_index * spixel_size[0] + spixel_size[0] / 2.0
            spixel_y_center = y_index * spixel_size[1] + spixel_size[1] / 2.0
            
            for mask in masks:
            
    
    for mask in masks:
        
    

class DataSet():
    def __init__(self, data_path, 
                 mode='train', valid_split=0.1,
                 size=[112, 112],
                 scale=[-0.5, 0.5],
                 pads=[0.1, 0.1],
                 flip_x=True,
                 flip_y=True,
                 swap_xy=True):
        self.mode = mode
        self.valid_split = valid_split
        self.data_path = data_path
        self.scale = scale
        self.pads = pads
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.swap_xy = swap_xy
        self.size = size
        
        dataset_file = os.path.join(data_path, 'train_dataset.json')
        
        fin = open(dataset_file)
        self.dataset = json.load(fin)
        fin.close()
        
        keys = sorted(list(self.dataset.keys()))
        
        n_valid = int(valid_split * len(keys))
        
        self.valid_keys = keys[:n_valid]
        self.train_keys = keys[n_valid:]
        self.rle = utils.RLECoder()
        
    def __len__(self):
        if self.mode == 'valid':
            return len(self.valid_keys)
        else:
            return len(self.train_keys)

    def __getitem__(self, index):
        keys = self.train_keys
        
        if self.mode == 'valid':
            keys = self.valid_keys
            
        key = keys[index]
        
        img_path = os.path.join(self.data_path, 'train', key, 'images', key + '.png')
        img = skimage.io.imread(img_path)
        
        img_shape = img.shape[:2]
        
        masks = deepcopy(self.dataset[key])
        
        for mask in masks:
            mask_tensor = self.rle.decode(mask[1])
            mask[1] = mask_tensor
        
        # JITTERING THE IMAGE
        
        top_pad = random.random() * self.pads[1]
        left_pad = random.random() * self.pads[0]
        bottom_pad = random.random() * self.pads[1]
        right_pad = random.random() * self.pads[0]
        
        img, masks = pad_image(img, masks, [left_pad, top_pad, right_pad, bottom_pad])
        
        img = skimage.transform.resize(img, img_shape, mode='reflect')
        
        scale_factor = numpy.exp(random.random() * (self.scale[1] - self.scale[0]) + self.scale[0])
        img = skimage.transform.rescale(img, scale_factor, mode='reflect')
        #print(img.shape)

        # CROPPING THE IMAGE
        
        left_pad = random.random() * (img.shape[0] - self.size[0] - 1) / (img.shape[0] - 1)
        right_pad = 1.0 - left_pad - (self.size[0] - 1) / (img.shape[0] - 1)
        top_pad = random.random() * (img.shape[1] - self.size[1] - 1) / (img.shape[1] - 1)
        bottom_pad = 1.0 - top_pad - (self.size[1] - 1) / (img.shape[1] - 1)
        
        img, masks = pad_image(img, masks, [left_pad, top_pad, right_pad, bottom_pad])
        
        masks = correct_masks(masks)
        
        # FLIPPING THE IMAGE
        
        flip_x_flag = False
        flip_y_flag = False
        swap_xy_flag = False
        
        if self.flip_x:
            flip_x_flag = (random.random() > 0.5)
        if self.flip_y:
            flip_y_flag = (random.random() > 0.5)
        if self.swap_xy:
            swap_xy_flag = (random.random() > 0.5)
            
        if flip_x_flag:
            img = numpy.flip(img, axis=0)
            for mask in masks:
                new_bbox = [1.0 - mask[0][2], mask[0][1], 
                            1.0 - mask[0][0], mask[0][3]]
                mask[0] = new_bbox
                
        if flip_y_flag:
            img = numpy.flip(img, axis=1)
            for mask in masks:
                new_bbox = [mask[0][0], 1.0 - mask[0][3], 
                            mask[0][2], 1.0 - mask[0][1]]
                mask[0] = new_bbox
                
        if swap_xy_flag:
            img = img.swapaxes(0, 1)
            for mask in masks:
                new_bbox = [mask[0][1], mask[0][0], 
                            mask[0][3], mask[0][2]]
                mask[0] = new_bbox
                
        for mask in masks:
            mask_tensor = mask[1]
            mask_tensor = skimage.transform.resize(
                mask_tensor, [18, 18], mode='reflect')
            
            if flip_x_flag:
                mask_tensor = numpy.flip(mask_tensor, axis=0).copy()
            if flip_y_flag:
                mask_tensor = numpy.flip(mask_tensor, axis=1).copy()
                
            if swap_xy_flag:
                mask_tensor = mask_tensor.swapaxes(0, 1)
                
            mask[1] = mask_tensor
        
        img = torch.from_numpy(img.copy()).transpose(0, 2)
        return [img], [masks]