import sys

sys.path.append('../src')

import os
import numpy
import skimage.io
import json
import utils

data_path = '../data'

rle_coder = utils.RLECoder()

masks = {}

masks_file = open(os.path.join(data_path, 'stage1_train_labels.csv'))
next(masks_file)

for line in masks_file:
    img_name, mask = line.split(',')
    mask = [int(x) for x in mask.split()]
    
    
    
    if img_name not in masks:
        masks[img_name] = []
        
    masks[img_name].append(mask)

data_path = os.path.join(data_path, 'train')
index = 0

new_masks = {}

all_images = os.listdir(data_path)

for index in range(len(all_images)):
    img_name = all_images[index]
    
    print(index, "out of ", len(all_images))
    
    image_path = os.path.join(data_path, img_name, 'images', img_name + '.png')
    img = skimage.io.imread(image_path)
    masks_path = os.path.join(data_path, img_name, 'masks')
    for mask in masks[img_name]:
        mask = rle_coder.decode(list(img.shape[1::-1]) + mask).T

        x_projection = mask.sum(axis=1)
        y_projection = mask.sum(axis=0)
                
        if x_projection.sum() > 0.5:
            left = 0
            right = 0
            while x_projection[left] < 0.5:
                left += 1
            
            for index in range(len(x_projection)):
                if x_projection[index] > 0.5:
                    right = index
                    
            top = 0
            bottom = 0
            
            while y_projection[top] < 0.5:
                top += 1
            
            for index in range(len(y_projection)):
                if y_projection[index] > 0.5:
                    bottom = index
            
            true_mask = mask[top:bottom + 1, left:right + 1 ]
            rle_mask = rle_coder.encode(true_mask)
            
            left = left / (img.shape[0] - 1)
            right = right / (img.shape[0] - 1)
            top = top / (img.shape[1] - 1)
            bottom = bottom / (img.shape[1] - 1)
        
            bbox = [left, top, right, bottom]
        
            if img_name not in new_masks:
                new_masks[img_name] = []
            new_masks[img_name].append([bbox, rle_mask])
        
fout = open('../data/train_dataset.json', 'w+')
json.dump(new_masks, fout)
fout.close()