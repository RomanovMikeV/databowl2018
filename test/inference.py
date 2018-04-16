import sys
sys.path.append('../src/')

import matplotlib as mpl
mpl.use('Agg')

import torch
import model
import skimage.io
import skimage.transform
import skimage.color
import numpy
import scipy
import os
import utils
#import 

# import the necessary packages
import numpy as np
 
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
    
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

from matplotlib import pyplot

rle = utils.RLECoder()
#checkpoint = torch.load('../models/model_bboxer.pth.tar')
model_boxer = torch.load('../train/stage2_checkpoint.pth.tar')

boxer_socket = model_boxer['socket']
boxer_socket.model.cuda()

model_masker = torch.load('../train/masker_checkpoint.pth.tar')

masker_socket = model_masker['socket']
masker_socket.model.cuda()

threshold = 0.05
iou_threshold = 0.3
indicator_threshold=0.5

image_bboxes = {}

test_data_folder = '../data/test_final'

results = {}
img_index = 0

for img_id in os.listdir(test_data_folder):
    img_index += 1
    
    if img_id not in results:
        results[img_id] = []
        
    image_bboxes[img_id] = []
    
    img_path = os.path.join(test_data_folder, img_id, 'images', img_id + '.png')
    original_img = skimage.io.imread(img_path, as_grey=True)
    original_img = skimage.color.grey2rgb(original_img)
    img_shape = original_img.shape[:-1]
    
    new_shape = [int(img_shape[0] / 32.0) * 32, int(img_shape[1] / 32.0) * 32]    
    #print(img.shape, '<- old shape')
    img = skimage.transform.resize(original_img, new_shape)
    img = skimage.color.rgb2grey(img)
    print(img_id, img_index, img.shape)
    
    if max(img.shape) > 700:
        continue
    #print(img.mean())
    
    if img.mean() > 0.5:
        img = 1.0 - img
    
    img = skimage.color.grey2rgb(img)
    #print(img.shape, '<- new shape')
    img = img.swapaxes(0, 2)
    
    
    
    img = torch.from_numpy(img).unsqueeze(0)
    
    
    img = img.float().cuda()
    #boxer_socket.model.cuda()
    predictions = boxer_socket.model.forward([img])
    #boxer_socket.model.cpu()
    img = img.cpu().numpy()[0].swapaxes(0, 2)
    
    predictions[0][:, 0:3, :, :] = torch.sigmoid(predictions[0][:, 0:3, :, :])
    predictions[0][:, 3:, :, :] = torch.exp(predictions[0][:, 3:, :, :])
    
    predictions[0] = predictions[0][0, :, :, :].detach().cpu().numpy()
    predictions[1] = predictions[1][0, :, :, :].detach().cpu().numpy()
    
    #pyplot.figure(figsize=(10.0, 10.0))
    #pyplot.imshow(original_img)
    
    x_indice, y_indice = numpy.where(predictions[0][0, :, :] > threshold)
    
    current_bboxes = []
    
    for bbox_index in range(len(x_indice)):
        x_index = x_indice[bbox_index]
        y_index = y_indice[bbox_index]
        
        prob    = predictions[0][0, x_index, y_index]
        
        x_shift = (predictions[0][1, x_index, y_index] - 0.5) * 2.0
        y_shift = (predictions[0][2, x_index, y_index] - 0.5) * 2.0
        
        width   = predictions[0][3, x_index, y_index]
        height  = predictions[0][4, x_index, y_index]
        
        mask    = predictions[1][:, x_index, y_index].reshape([15, 15])
        
        x_center = (x_index + 0.5 + x_shift) * 4.0
        y_center = (y_index + 0.5 + y_shift) * 4.0
        
        width = width * 4.0
        height = height * 4.0
        
        left = x_center - height / 2.0
        right = x_center +  height / 2.0
        top = y_center - width / 2.0
        bottom = y_center + width / 2.0
        
        current_bboxes.append([left, top, right, bottom])
    
    current_bboxes = numpy.array(current_bboxes)
    del predictions[1]
    del predictions[0]
    
    current_bboxes = non_max_suppression_fast(current_bboxes, iou_threshold)
    
    if len(current_bboxes) == 0:
        continue
    
    crops = []
    masks = []
    non_zero_boxes = []
    indicators = []
    
    for index in range(current_bboxes.shape[0]):
        left = current_bboxes[index, 0]
        top = current_bboxes[index, 1]
        right = current_bboxes[index, 2]
        bottom = current_bboxes[index, 3]
        
        x_center = (left + right) / 2.0
        y_center = (top + bottom) / 2.0
        
        width = right - left
        height = bottom - top
        
        new_width = width * 1.1 * 3.0
        new_height = height * 1.1 * 3.0
        
        new_right = int(round(x_center + new_width / 2.0))
        new_left = int(round(x_center - new_width / 2.0))
        new_top = int(round(y_center - new_height / 2.0))
        new_bottom = int(round(y_center + new_height / 2.0))
        
        #print(left, right, top, bottom)
        left = left / new_shape[1] * img_shape[1]
        right = right / new_shape[1] * img_shape[1]
        top = top / new_shape[0] * img_shape[0]
        bottom = bottom / new_shape[0] * img_shape[0]
        
        #masker_socket.model.cuda()
        if right > left and bottom > top:
            crop = cutout(img, [new_top, new_bottom], [new_left, new_right])
            crop = skimage.transform.resize(crop, [96, 96])
            
            crops.append(crop)
            non_zero_boxes.append([left, right, top, bottom])
            
            crop = torch.from_numpy(crop.swapaxes(2, 1).swapaxes(1, 0)).unsqueeze(0)
            output = masker_socket.model.forward([crop.float().cuda()])
            mask = output[0].cpu().data.numpy()[0, 0, 32:64, 32:64]
            indicator = output[1].cpu().data.item()
            
            del output[1]
            del output[0]
            
            indicators.append(indicator)
            masks.append(mask)
            
            
        #masker_socket.model.cpu()
        

        
        #pyplot.plot([left, right], [top, top], '--r')
        #pyplot.plot([left, right], [bottom, bottom], '--r')
        #pyplot.plot([left, left], [top, bottom], '--r')
        #pyplot.plot([right, right], [top, bottom], '--r')
    

        
    
    all_masks = numpy.zeros(original_img.shape)
    
    for index in range(len(crops)):
        #pyplot.figure(figsize=(10.0, 10.0))
        
        indicator = indicators[index]
        crop = crops[index][32:64, 32:64, :]
        mask = masks[index]
        box = non_zero_boxes[index]
        
        left = max(int(box[0]), 0)
        right = min(int(box[1]), img_shape[1])
        top = max(int(box[2]), 0)
        bottom = min(int(box[3]), img_shape[0])
        
        
        if indicator > indicator_threshold and top < bottom and left < right:
            #print(left, right, top, bottom)
            mask = skimage.transform.resize(mask, [bottom-top, right-left], mode='constant', order=3, cval=0)
            all_masks[top:bottom, left:right, 0] = (mask > 0.5).astype(float)
            one_mask = numpy.zeros(all_masks.shape[:2])
            one_mask[top:bottom, left:right] = (mask > 0.5)
            
            rle_mask = rle.encode(one_mask)
            results[img_id].append(rle_mask)
            #results.append([img_id, rle_mask])
            
            #print(indicator)
            #print(rle_mask)
            
            #pyplot.imshow(crop)
        
            #pyplot.savefig(os.path.join('results', 'crop_' + img_id + '_' + str(index) + '.jpg'))
            #pyplot.close('all')
    
    #print(all_masks)
    #pyplot.savefig(os.path.join('results', img_id + '.jpg'))
    #pyplot.close('all')
    
    #pyplot.figure(figsize=(10.0, 10.0))
    #original_img += all_masks.astype(float) * 0.1
    #pyplot.imshow(all_masks)
            
    #pyplot.savefig(os.path.join('results', img_id + '_masks.jpg'))
    #pyplot.close('all')
    
    #print(results[-1])
    #throw_error()
    
    #print(img.shape)
    #print(predictions[0].shape) 
    #print(predictions[1].shape)
    
    #break
    #print(img.shape)

with open('submission.csv', 'w+') as fout:
    fout.write("ImageId,EncodedPixels")
    for img_id in results:
        for mask in results[img_id]:
            fout.write(img_id + ',' + ' '.join([str(x) for x in mask]) + '\n')
        
print('Done')