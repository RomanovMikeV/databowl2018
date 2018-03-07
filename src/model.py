import torch
from torchvision import models
from torch import nn


class Network(nn.Module):
    def __init__(self, 
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
        else:
            pass
        
        self.up1 = torch.nn.Conv2d(2048, 4096, (3, 3), padding=(1, 1))
        self.up_bn1 = torch.nn.BatchNorm2d(4096)
        self.up2 = torch.nn.Conv2d(1024, 2048, (3, 3), padding=(1, 1))
        self.up_bn2 = torch.nn.BatchNorm2d(2048)
        self.up3 = torch.nn.Conv2d(512, 1024, (3, 3), padding=(1, 1))
        self.up_bn3 = torch.nn.BatchNorm2d(1024)
        
        self.ps = torch.nn.PixelShuffle(2)
        
        self.elu = torch.nn.ELU()
        
        self.detector = torch.nn.Conv2d(256, (5 + 81) * 3, (1, 1))
        
        
    
    def forward(self, input):
        
        x = input
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
        
        # Detecting objects (by now I have here )
        result = self.detector(layer_1_out + layer_2_up)

        return [result]




class Socket():
    def __init__(self, model):
        self.model = model
        self.loss_function = torch.nn.BCEWithLogitsLoss()
    
    def criterion(self, output, target):
        
        # loss for object detection
        # loss for bounding box misplacement
        # loss for bounding box width and height loss
        # loss for mask
        
        return self.loss_function(output[0], target[0])
    
    def metrics(self, outputs, targets):
        # object roc-auc
        # bbox intersection over union
        # mask intersection over union
        bce_loss = self.loss_function(outputs[0], targets[0])
        return [bce_loss]