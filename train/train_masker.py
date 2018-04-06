import sys

sys.path.append('../src')

import torch
import torchvision
import trainer
import dataset_masker as dataset
import skimage.transform
import model_masker as model
import skimage.filters.rank
import skimage.morphology

import utils

anchors = [[1.0, 1.0]]
mask_size = [15, 15]

train_dataset = dataset.DataSet('../data/', mode='train', anchors=anchors, 
                                mask_size=mask_size, size=[32, 32], surrounding=3.0)
valid_dataset = dataset.DataSet('../data/', mode='valid', anchors=anchors, mask_size=mask_size,
                                size=[32, 32], surrounding=3.0)

img, mask = train_dataset[10]


print(img[0].size())
print(mask[0].size())
# new_masks = numpy.zeros(img.shape)
# new_masks[32:64, 32:64, 0] = mask
# imshow(img)
# imshow(new_masks, alpha=0.1)
# imshow(mask)

# for stress_index in range(100):
#     for index in range(len(train_dataset)):
#         img, mask = train_dataset[5]
#     print(stress_index, end=',')

# def nms(image, bboxes):

# imshow(img)
# bbox = mask[0]
# left = round(bbox[0] * (img.shape[0] - 1))
# top = round(bbox[1] * (img.shape[1] - 1))
# right = round(bbox[2] * (img.shape[0] - 1))
# bottom = round(bbox[3] * (img.shape[1] - 1))

# plot([top, bottom], [left, right])

# imshow(mask)

net = model.Network(len(anchors), mask_size, ).float().cuda()
input = torch.autograd.Variable(torch.zeros(10, 3, 128, 128).float().cuda())
result = net.forward([input])

socket = model.Socket(net)

pretrain_modules = torch.nn.ModuleList([socket.model.up_bn1, 
                                        socket.model.up_bn2, 
                                        socket.model.up_bn3,
                                        socket.model.up_bn4,
                                        socket.model.up_bn5,
                                        socket.model.up1, 
                                        socket.model.up2, 
                                        socket.model.up3,
                                        socket.model.up4,
                                        socket.model.up5,
                                        socket.model.masker,
                                        socket.model.decider]).parameters()
# train_moduels = net.parameters()

pretrain_optimizer = torch.optim.Adam(pretrain_modules, lr=3.0e-5)
my_pretrainer = trainer.Trainer(socket, pretrain_optimizer)


train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=4, 
                                           shuffle=True,
                                           num_workers=2,
                                           drop_last=True,)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=2,
                                           shuffle=True,
                                           num_workers=2,
                                           drop_last=False,)

for epoch in range(1000):
    print("Epoch", epoch)
    print("Training ... ", end='')
    
    loss = my_pretrainer.train(train_loader)
    print("Done")
    print("Validating on train ... ", end='')
    train_metrics = my_pretrainer.validate(train_loader)
    print("Done")
    print("Validating on valid ... ", end='')
    valid_metrics = my_pretrainer.validate(valid_loader)
    print("Done")
    print('; '.join([str(x) for x in train_metrics + valid_metrics]))
    
#     my_pretrainer.validate()
    if epoch % 100 == 0:
        my_pretrainer.make_checkpoint(prefix='masker_pretraining_')

train_modules = socket.model.parameters()

train_optimizer = torch.optim.Adam(train_modules, lr=3.0e-5)
my_trainer = trainer.Trainer(socket, train_optimizer)

for epoch in range(10000):
    print("Epoch", epoch)
    print("Training ... ", end='')
    
    loss = my_pretrainer.train(train_loader)
    print("Done")
    print("Validating ... ", end='')
    train_metrics = my_pretrainer.validate(train_loader)
    valid_metrics = my_pretrainer.validate(valid_loader)
    print("Done")
    print('; '.join([str(x) for x in train_metrics + valid_metrics]))

    if epoch % 100 == 0:
        my_trainer.make_checkpoint(prefix='masker_')