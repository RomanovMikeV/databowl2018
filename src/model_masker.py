import torch
from torchvision import models
from torch import nn
import sklearn.metrics
import numpy
import scipy.special

class Network(nn.Module):
    def __init__(self,
                 n_anchors,
                 mask_size,
                 arch='resnet50',
                 n_inputs=2,
                 n_outputs=1,
                 pretrained=True,
                 incline=True):
        
        super(Network, self).__init__()
        
        if arch == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)

        elif arch == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        
        elif arch == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        
        elif arch == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        
        elif arch == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
            
        else:
            pass
        
        self.up1 = torch.nn.Conv2d(2048, 2048 * 2, (3, 3), padding=(1, 1))
        self.up_bn1 = torch.nn.BatchNorm2d(4096)
        self.up2 = torch.nn.Conv2d(2048, 2048, (3, 3), padding=(1, 1))
        self.up_bn2 = torch.nn.BatchNorm2d(2048)
        self.up3 = torch.nn.Conv2d(1024, 1024, (3, 3), padding=(1, 1))
        self.up_bn3 = torch.nn.BatchNorm2d(1024)
        self.up4 = torch.nn.Conv2d(576, 512, (3, 3), padding=(1, 1))
        self.up_bn4 = torch.nn.BatchNorm2d(512)
        self.up5 = torch.nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.up_bn5 = torch.nn.BatchNorm2d(256)
        
        self.mask_size = mask_size
        
        self.ps = torch.nn.PixelShuffle(2)
        
        self.elu = torch.nn.ELU()
        
        self.masker = torch.nn.Conv2d(67, 1, (1, 1))
        
        
    
    def forward(self, input):
        
        layer_input = input[0]
        
        x = self.model.conv1(layer_input)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        layer_0_out = x
        
        # Scaling down
        layer_1_out = self.model.layer1(layer_0_out)
        layer_2_out = self.model.layer2(layer_1_out)
        layer_3_out = self.model.layer3(layer_2_out)
        layer_4_out = self.model.layer4(layer_3_out)
        
        # Scaling up
        layer_4_up = self.elu(self.ps(self.up_bn1(self.up1(layer_4_out))))
        
        #print(layer_4_up.size(), '<- layer 4 up')
        
        layer_3_up = self.elu(
            self.ps(self.up_bn2(self.up2(torch.cat([layer_3_out, layer_4_up], dim=1)))))
        
        #print(layer_3_up.size(), '<- layer 3 up')
        #print(layer_2_out.size(), '<- layer 2 out')
        #print(self.up3)
        
        layer_2_up = self.elu(
            self.ps(self.up_bn3(self.up3(torch.cat([layer_3_up, layer_2_out], dim=1)))))
        
        #print(layer_2_up.size(), '<- layer 2 up')
        
        layer_1_up = self.elu(
            self.ps(self.up_bn4(self.up4(torch.cat([layer_2_up, layer_1_out, layer_0_out], dim=1)))))
        
        #print(layer_1_up.size(), '<- layer 1 up')
        #print(layer_0_out.size(), '<- layer 0 out')
        
        layer_0_up = self.elu(
            self.ps(self.up_bn5(self.up5(layer_1_up))))
        
        #print(layer_0_up.size(), '<- layer 0 up size')
        
        result = torch.sigmoid(self.masker(torch.cat([layer_0_up, layer_input], dim=1)))
        
        #print(result.size(), '<- result size')
        
        
        # Detecting objects
        #result = self.detector(layer_1_out + layer_2_up)
        
        
        #masks = torch.sigmoid(self.masker(layer_1_out + layer_2_up))

        return [result]


class Socket():
    def __init__(self, model):
        self.model = model
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        
    @staticmethod
    def binary_ce(outputs, targets):
        return - (targets * torch.log(outputs + 1.0e-8) + 
                  (1.0 - targets) * torch.log(1.0 - outputs + 1.0e-8)).mean()
    
    
    def criterion(self, output, target):
        return self.binary_ce(output[0][:, :, 32:64, 32:64], target[0])

    
    @staticmethod
    def mask_iou(predicted, target):
        
        predicted = predicted[:, :, 32:64, 32:64]
        
        intersections = (predicted * target).sum(dim=3).sum(dim=2).sum(dim=1)
        unions = predicted.sum(dim=3).sum(dim=2).sum(dim=1) + target.sum(dim=3).sum(dim=2).sum(dim=1) - intersections
        
        return (intersections / (unions + 1.0e-8)).sum() / target.size(0)
    
    
    def metrics(self, outputs, targets):
        results = []
        
        masks_iou = self.mask_iou((outputs[0] > 0.5).float(), targets[0]).item()
            
        results.append(masks_iou)
        return results