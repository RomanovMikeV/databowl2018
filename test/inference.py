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
#import 

from matplotlib import pyplot

#checkpoint = torch.load('../models/model_bboxer.pth.tar')
checkpoint = torch.load('../train/checkpoint.pth.tar')

socket = checkpoint['socket']
threshold = 0.05

test_data_folder = '../data/test'
for img_id in os.listdir(test_data_folder):
    img_path = os.path.join(test_data_folder, img_id, 'images', img_id + '.png')
    img = skimage.io.imread(img_path)[:, :, :3]
    img_shape = img.shape[1:]
    
    new_shape = [int(img.shape[0] / 32.0) * 32, int(img.shape[1] / 32.0) * 32]    
    #print(img.shape, '<- old shape')
    img = skimage.transform.resize(img, new_shape)
    img = skimage.color.rgb2grey(img)
    print(img_id)
    print(img.mean())
    
    if img.mean() > 0.5:
        img = 1.0 - img
    
    img = skimage.color.grey2rgb(img)
    #print(img.shape, '<- new shape')
    img = img.swapaxes(0, 2)
    
    
    
    img = torch.from_numpy(img).unsqueeze(0)
    
    
    img = img.float().cuda()
    predictions = socket.model.forward([img])
    img = img.cpu().numpy()[0].swapaxes(0, 2)
    
    predictions[0][:, 0:3, :, :] = torch.sigmoid(predictions[0][:, 0:3, :, :])
    predictions[0][:, 3:, :, :] = torch.exp(predictions[0][:, 3:, :, :])
    
    predictions[0] = predictions[0][0, :, :, :].detach().cpu().numpy()
    predictions[1] = predictions[1][0, :, :, :].detach().cpu().numpy()
    
    pyplot.figure(figsize=(10.0, 10.0))
    pyplot.imshow(img)
    
    x_indice, y_indice = numpy.where(predictions[0][0, :, :] > threshold)
    
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
        
        pyplot.plot([left, right], [top, top], '-r')
        pyplot.plot([left, right], [bottom, bottom], '-r')
        pyplot.plot([left, left], [top, bottom], '-r')
        pyplot.plot([right, right], [top, bottom], '-r')
        
    
    
    pyplot.savefig(os.path.join('results', img_id + '.jpg'))
    pyplot.close('all')
    
    print(img.shape)
    print(predictions[0].shape) 
    print(predictions[1].shape)
    
    print(img.shape)

print('Done')