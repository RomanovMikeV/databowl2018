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

def inverse_sigmoid(x):
    res = - numpy.log(2.0 / (x + 1.0) - 1.0)
    return res

def masks_to_targets(masks, 
                     anchors=[[1.0, 1.0]], 
                     mask_shape=[9, 9], 
                     img_size=[128, 128],
                     spixel_size=[4.0, 4.0]):
    
    anchor_size = 5
    mask_size = mask_shape[0] * mask_shape[1]

    targets_shape = [int(img_size[0] / spixel_size[0]),
                     int(img_size[1] / spixel_size[1]),
                     anchor_size * len(anchors)]
    masks_shape = [targets_shape[0],
                   targets_shape[1],
                   len(anchors) * mask_size]
    
    targets = numpy.zeros(targets_shape)
    targets_masks = numpy.zeros(masks_shape)
    
    for index in range(len(masks)):
        x_center = (masks[index][0][0] + masks[index][0][2]) / 2.0
        y_center = (masks[index][0][1] + masks[index][0][3]) / 2.0
        
        x_position = x_center * img_size[0] / spixel_size[0] - 0.5
        y_position = y_center * img_size[1] / spixel_size[1] - 0.5
        
        x_index = round(x_position)
        y_index = round(y_position)
        
        x_shift = x_position - x_index
        y_shift = y_position - y_index
        
        width = (masks[index][0][2] - masks[index][0][0]) * (
            img_size[0] / spixel_size[0])
        height = (masks[index][0][3] - masks[index][0][1]) * (
            img_size[1] / spixel_size[1])
        
        best_iou = 0.0
        best_anchor_index = None
        
        for anchor_index in range(len(anchors)):
            anchor = anchors[anchor_index]
            
            c_width = min(width, anchor[0])
            c_height = min(height, anchor[1])
            
            intersection = c_width * c_height
            union = width * height + anchor[0] * anchor[1] - intersection
            
            iou = intersection / union
        
            if best_iou < iou:
                
                best_iou = iou
                best_anchor_index = anchor_index
    
        best_anchor = anchors[anchor_index]
    
        if targets[x_index, y_index, best_anchor_index * (5 + mask_shape[0] * mask_shape[1]) + 0] > 0.5:
            print("Conflict", best_anchor_index)
            
        targets[x_index, y_index, best_anchor_index * anchor_size + 0] = 1.0
        targets[x_index, y_index, best_anchor_index * anchor_size + 1] = inverse_sigmoid(x_shift)
        targets[x_index, y_index, best_anchor_index * anchor_size + 2] = inverse_sigmoid(y_shift)
        targets[x_index, y_index, best_anchor_index * anchor_size + 3] = numpy.log(width  / best_anchor[0])
        targets[x_index, y_index, best_anchor_index * anchor_size + 4] = numpy.log(height / best_anchor[1])
        
        targets_masks[x_index, y_index, :] = masks[index][1].flatten()
        
    return targets, targets_masks

def cutout(array, x_cut, y_cut):
    x_start = x_cut[0]
    x_end   = x_cut[1]
    
    y_start = y_cut[0]
    y_end   = y_cut[1]
    
    pad_left   = 0
    pad_right  = 0
    pad_top    = 0
    pad_bottom = 0
    
    if x_start < 0:
        pad_left = -x_start
        x_start = 0
        
    if y_start < 0:
        pad_top = -y_start
        y_start = 0
        
    if x_end > array.shape[0]:
        pad_right = x_end - array.shape[0]
        x_end = array.shape[0]
        
    if y_end > array.shape[1]:
        pad_bottom = y_end - array.shape[1]
        y_end = array.shape[1]
        
    res = array[x_start:x_end, y_start:y_end, :3]
    
    #print([x_start, x_end], [y_start, y_end])
    #print(res.shape)
    #print([pad_left, pad_right], [pad_top, pad_bottom])
    
    res = numpy.pad(res, [[pad_left, pad_right], 
                          [pad_top, pad_bottom],
                          [0, 0]], 
                         mode='constant', constant_values=0)
    
    return res

class DataSet():
    def __init__(self, data_path, 
                 mode='train', valid_split=0.1,
                 anchors=[[1.0, 1.0]],
                 mask_size=[9, 9],
                 size=[128, 128],
                 scale=[-0.5, 0.5],
                 min_pads=[0.1, 0.1],
                 max_pads=[0.3, 0.3],
                 surrounding=3.0,
                 flip_x=True,
                 flip_y=True,
                 swap_xy=True):
        self.mode = mode
        self.valid_split = valid_split
        self.data_path = data_path
        self.scale = scale
        self.min_pads = min_pads
        self.max_pads = max_pads
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.swap_xy = swap_xy
        self.size = size
        self.anchors = anchors
        self.mask_size=mask_size
        self.surrounding = surrounding
        
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
        img = skimage.color.rgb2grey(img)
        
        if img.mean() > 0.5:
            img = 1.0 - img
    
        img = skimage.color.grey2rgb(img)
        
        img_shape = img.shape[:2]
        
        masks = deepcopy(self.dataset[key])
        
        # Here I select a random mask
        
        mask = masks[random.randrange(len(masks))]
        
        mask_tensor = self.rle.decode(mask[1])
        mask[1] = mask_tensor
        #print(mask[1])
        
        #print(mask[0], mask[1].shape)
        bbox = mask[0]
        
        left = round(bbox[0] * (img.shape[0]))
        top = round(bbox[1] * (img.shape[1]))
        right = round(bbox[2] * (img.shape[0]))
        bottom = round(bbox[3] * (img.shape[1]))
        
        width = right - left
        height = bottom - top
        
        pad_left   = int(width * random.uniform(self.min_pads[0], self.max_pads[0]))
        pad_right  = int(width * random.uniform(self.min_pads[0], self.max_pads[0]))
        
        pad_top    = int(height * random.uniform(self.min_pads[1], self.max_pads[1]))
        pad_bottom = int(height * random.uniform(self.min_pads[1], self.max_pads[1]))
        
        pad_left, pad_top, pad_right, pad_bottom = 0, 0, 0, 0
        
        new_left = left - pad_left
        new_right = right + pad_right
        new_top = top - pad_top
        new_bottom = bottom + pad_bottom
        
        center_x = (new_left + new_right) / 2.0
        center_y = (new_top + new_bottom) / 2.0
        
        new_width = (new_right - new_left) / 2.0 * self.surrounding
        new_height = (new_bottom - new_top) / 2.0 * self.surrounding
        
        new_left = int(center_x - new_width)
        new_right = int(center_x + new_width) + 1
        
        new_top = int(center_y - new_height)
        new_bottom = int(center_y + new_height) + 1
        
        #print([new_left, new_right], [new_top, new_bottom])
        
        img = cutout(img, [new_left, new_right], [new_top, new_bottom])
        
        mask = mask[1]
        
        #print([pad_left, pad_right], [pad_top, pad_bottom])
        mask = numpy.pad(mask, 
                         ((pad_right, pad_left), 
                          (pad_bottom, pad_top)), 
                         mode='constant', constant_values=0)
        
        img_size = []
        for size in self.size:
            img_size.append(size * self.surrounding)
        
        img = skimage.transform.resize(img, img_size, mode='constant', order=3, cval=0)
        mask = skimage.transform.resize(mask, self.size, mode='constant', order=3, cval=0)
        
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
            mask = numpy.flip(mask, axis=0)
                
        if flip_y_flag:
            img = numpy.flip(img, axis=1)
            mask = numpy.flip(mask, axis=1)
                
        if swap_xy_flag:
            img = img.swapaxes(0, 1)
            mask = mask.swapaxes(0, 1)
        
        img = torch.Tensor(img.copy()).transpose(0, 2)
        mask = torch.Tensor(mask.copy()).unsqueeze(-1).transpose(0, 2)
        return [img], [mask]
        
        '''
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
                mask_tensor, self.mask_size, mode='reflect')
            
            if flip_x_flag:
                mask_tensor = numpy.flip(mask_tensor, axis=0).copy()
            if flip_y_flag:
                mask_tensor = numpy.flip(mask_tensor, axis=1).copy()
                
            if swap_xy_flag:
                mask_tensor = mask_tensor.swapaxes(0, 1)
                
            mask[1] = mask_tensor
        
        img = torch.from_numpy(img.copy()).transpose(0, 2).float()
        
        targets_bboxes, targets_masks = masks_to_targets(
                             masks, 
                             anchors=self.anchors,
                             mask_shape=self.mask_size, 
                             img_size=self.size,
                             spixel_size=[4, 4])
        
        targets_bboxes = torch.from_numpy(targets_bboxes).transpose(0, 2).float()
        targets_masks = torch.from_numpy(targets_masks).transpose(0, 2).float()
        
        return [img[:3, :, :]], [targets_bboxes, targets_masks]
        '''