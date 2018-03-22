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
        self.up2 = torch.nn.Conv2d(1024, 2048, (3, 3), padding=(1, 1))
        self.up_bn2 = torch.nn.BatchNorm2d(2048)
        self.up3 = torch.nn.Conv2d(512, 1024, (3, 3), padding=(1, 1))
        self.up_bn3 = torch.nn.BatchNorm2d(1024)
        self.mask_size = mask_size
        
        self.ps = torch.nn.PixelShuffle(2)
        
        self.elu = torch.nn.ELU()
        
        self.detector = torch.nn.Conv2d(256, (5) * n_anchors, (1, 1))
        self.masker = torch.nn.Conv2d(256, self.mask_size[0] * self.mask_size[1] * n_anchors, (1, 1))
        
        
    
    def forward(self, input):
        
        x = input[0]
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        # Scaling down
        layer_1_out = self.model.layer1(x)
        layer_2_out = self.model.layer2(layer_1_out)
        layer_3_out = self.model.layer3(layer_2_out)
        layer_4_out = self.model.layer4(layer_3_out)
        
        # Scaling up
        layer_4_up = self.elu(self.ps(self.up_bn1(self.up1(layer_4_out))))
        layer_3_up = self.elu(
            self.ps(self.up_bn2(self.up2(layer_3_out + layer_4_up))))
        layer_2_up = self.elu(
            self.ps(self.up_bn3(self.up3(layer_3_up + layer_2_out))))
        
        # Detecting objects
        result = self.detector(layer_1_out + layer_2_up)
        masks = torch.sigmoid(self.masker(layer_1_out + layer_2_up))

        return [result, masks]


class Socket():
    def __init__(self, model, anchors,
                 penalize_bbox=True,
                 penalize_mask=True):
        self.model = model
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.penalize_bbox = penalize_bbox
        self.penalize_mask = penalize_mask
        self.anchors = anchors
        
    @staticmethod
    def binary_ce(outputs, targets):
        return - (targets * torch.log(outputs + 1.0e-8) + 
                  (1.0 - targets) * torch.log(1.0 - outputs + 1.0e-8))
    
    
    def criterion(self, output, target):
        
        # loss for object detection
        # loss for bounding box misplacement
        # loss for bounding box width and height loss
        # loss for mask
        object_loss = 0.0
        bbox_loss = 0.0
        mask_loss = 0.0
        
        anchor_size = 5
        mask_size = self.model.mask_size[0] * self.model.mask_size[1]
        
        for anchor_index in range(len(self.anchors)):
            object_loss = object_loss + (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    output[0][:, anchor_index * 5 + 0, :, :], 
                    target[0][:, anchor_index * 5 + 0, :, :]))
        
        
        
            bbox_loss = bbox_loss + (
                target[0][:, anchor_index * anchor_size + 0, :, :] * (
                torch.abs(output[0][:, anchor_index * anchor_size + 1, :, :] - 
                          target[0][:, anchor_index * anchor_size + 1, :, :]) +
                torch.abs(output[0][:, anchor_index * anchor_size + 2, :, :] - 
                          target[0][:, anchor_index * anchor_size + 2, :, :]) +
                torch.abs(output[0][:, anchor_index * anchor_size + 3, :, :] - 
                          target[0][:, anchor_index * anchor_size + 3, :, :]) + 
                torch.abs(output[0][:, anchor_index * anchor_size + 4, :, :] - 
                          target[0][:, anchor_index * anchor_size + 4, :, :])
                )).sum()
            
            #print(output[1].shape)
            #print(target[1].shape)
            #print(anchor_index * mask_size, (anchor_index + 1) * mask_size)
            
            mask_loss = mask_loss + (
                target[0][:, anchor_index * anchor_size, :, :] * (
                    self.binary_ce(
                        output[1][:, anchor_index * mask_size:(anchor_index + 1) * mask_size, :, :], 
                        target[1][:, anchor_index * mask_size:(anchor_index + 1) * mask_size, :, :]).mean(dim=1)
                )
            ).sum()
        
        return object_loss + bbox_loss + mask_loss
    
    @staticmethod
    def get_bboxes(tensor, anchors):
        
        result = tensor.data.numpy()
        
        for anchor_index in range(len(anchors)):
            
            anchor_position = anchor_index * 5
            result[:, anchor_position + 0, :, :] = result[:, anchor_position + 0, :, :]
            
            result[:, anchor_position + 1, :, :] = 2.0 * scipy.special.expit(
                tensor[:, anchor_position + 1, :, :]) - 1.0
            result[:, anchor_position + 2, :, :] = 2.0 * scipy.special.expit(
                result[:, anchor_position + 2, :, :]) - 1.0
            
            result[:, anchor_position + 3, :, :] = numpy.exp(
                result[:, anchor_position + 3, :, :]) * anchors[anchor_index][0]
            result[:, anchor_position + 4, :, :] = numpy.exp(
                result[:, anchor_position + 4, :, :]) * anchors[anchor_index][1]
            
        return result
    
    @staticmethod    
    def bboxes_iou(predicted, target):
        
        ious = 0.0
        counts = 0.0
        
        for anchor_position in range(0, predicted.shape[1], 5):
            intersections = \
                numpy.maximum(numpy.minimum(
                    predicted[:, anchor_position + 1, :, :] + predicted[:, anchor_position + 3, :, :] / 2,
                    target[:, anchor_position + 1, :, :] + target[:, anchor_position + 3, :, :] / 2) - 
                 numpy.maximum(
                    predicted[:, anchor_position + 1, :, :] - predicted[:, anchor_position + 3, :, :] / 2,
                    target[:, anchor_position + 1, :, :] - target[:, anchor_position + 3, :, :] / 2), 0) * \
                numpy.maximum(numpy.minimum(
                    predicted[:, anchor_position + 2, :, :] + predicted[:, anchor_position + 4, :, :] / 2,
                    target[:, anchor_position + 2, :, :] + target[:, anchor_position + 4, :, :] / 2) - 
                numpy.maximum(
                    predicted[:, anchor_position + 2, :, :] - predicted[:, anchor_position + 4, :, :] / 2,
                    target[:, anchor_position + 2, :, :] - target[:, anchor_position + 4, :, :] / 2), 0)
                
            unions = (
                predicted[:, anchor_position + 3, :, :] * predicted[:, anchor_position + 4, :, :] + 
                target[:, anchor_position + 3, :, :] * target[:, anchor_position + 4, :, :] -
                intersections)
            
            ious += (intersections / unions * target[:, anchor_position + 0, :, :]).sum()
            counts += target[:, anchor_position + 0, :, :].sum()
        
        return ious / counts
    
    
    @staticmethod
    def mask_iou(predicted, target, mask_size):
        
        ious = 0.0
        counts = 0.0
        
        for anchor_index in range(0, int(predicted.shape[1] / mask_size[0] / mask_size[1])):
            mask_start = anchor_index * mask_size[0] * mask_size[1]
            mask_end = mask_start + mask_size[0] * mask_size[1]
            
            predicted_masks = predicted[:, mask_start:mask_end, :, :]
            target_masks = target[:, mask_start:mask_end, :, :]
            indicators = (target[:, mask_start:mask_end, :, :].sum(dim=1) > 0.1).float()
            
            intersection = (predicted_masks * target_masks).sum(dim=1)
            union = predicted_masks.sum(dim=1) + target_masks.sum(dim=1) - intersection
            
            ious += (intersection / (union + 1.0e-8) * indicators).sum()
            counts += indicators.sum()

        return ious / counts
        
        
    
    
    def metrics(self, outputs, targets):
        # object roc-auc
        # bbox intersection over union
        # mask intersection over union
        # print(outputs[0].size())
        # print(targets[0].size())
        
        results = []
        
        prep_outputs = self.get_bboxes(outputs[0], self.anchors)
        prep_targets = self.get_bboxes(targets[0], self.anchors)
        
        for anchor_position in range(0, targets[0].size(1), 5):

            obj_roc_auc = sklearn.metrics.roc_auc_score(
                prep_targets[:, anchor_position + 0, :, :].flatten(), 
                prep_outputs[:, anchor_position + 0, :, :].flatten())
            
            bboxes_iou = self.bboxes_iou(prep_outputs, prep_targets)
            masks_iou = self.mask_iou((outputs[1] > 0.5).float(), targets[1], self.model.mask_size).item()
            
            results.append(obj_roc_auc)
            results.append(bboxes_iou)
            results.append(masks_iou)
        #n_bboxes = 0
        #bbox_iou = 0.0
        
        #obj_targets = targets[:, 0, :, :].data.numpy()
        
        #if n_bboxes > 0:
        #    bbox_iou = bbox_iou / n_bboxes
            
        
        #bce_loss = self.loss_function(outputs[0], targets[0])
        return results