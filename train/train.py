import sys

sys.path.append('../src')

import torch
import torchvision
import trainer
import dataset
import skimage.transform
import model

import utils

anchors = [[1.0, 1.0]]
mask_size = [15, 15]

train_dataset = dataset.DataSet('../data/', 
                                mode='train', 
                                anchors=anchors, 
                                mask_size=mask_size,
                                size=[224, 224])
valid_dataset = dataset.DataSet('../data/', 
                                mode='valid', 
                                anchors=anchors, 
                                mask_size=mask_size,
                                size=[224, 224])

# Basic collator function
def my_collator(args):
    res = None
    for arg in args:
        if res is None:
            res = []
            for item in arg:
                res.append([])
        for index in range(len(arg)):
            res[index].append(arg[index])
    
    return res

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=4, 
                                           shuffle=True,
                                           num_workers=2,
                                           drop_last=True,)
#                                            collate_fn=my_collator)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=4,
                                           shuffle=True,
                                           num_workers=2,
                                           drop_last=False,)
#                                            collate_fn=my_collator)

net = model.Network(len(anchors), mask_size).float().cuda()
input = torch.autograd.Variable(torch.zeros(10, 3, 128, 128).float().cuda())
result = net.forward([input])


socket = model.Socket(net, anchors)
'''
pretrain_modules = torch.nn.ModuleList([socket.model.up_bn1, 
                                        socket.model.up_bn2, 
                                        socket.model.up_bn3,
                                        socket.model.up1, 
                                        socket.model.up2, 
                                        socket.model.up3,
                                        socket.model.detector]).parameters()
# train_moduels = net.parameters()
'''

'''
pretrain_optimizer = torch.optim.Adam(pretrain_modules, lr=3.0e-4)
my_pretrainer = trainer.Trainer(socket, pretrain_optimizer)

for index in range(1000):
#     print("Epoch", index)
    print("Epoch", index)
    print("Training ... ", end='')
    loss = my_pretrainer.train(train_loader)
    print("Done")
    print("Validating ... ", end='')
    train_metrics = my_pretrainer.validate(train_loader)
    valid_metrics = my_pretrainer.validate(valid_loader)
    print("Done")
    print('; '.join([str(x) for x in train_metrics + valid_metrics]))
    
#     my_pretrainer.validate()

my_pretrainer.make_checkpoint('checkpoint_pretrained.t7')
'''

train_modules = socket.model.parameters()

train_optimizer = torch.optim.Adam(train_modules, lr=3.0e-4)
my_trainer = trainer.Trainer(socket, train_optimizer)
my_trainer.load_checkpoint('checkpoint.pth.tar')

socket = my_trainer.socket
train_modules = socket.model.parameters()
train_optimizer = torch.optim.Adam(train_modules, lr=3.0e-6)
my_trainer.optimizer = train_optimizer

for index in range(1000):
#     print("Epoch", index)
    print("Epoch", index)
    print("Training ... ", end='')
    loss = my_trainer.train(train_loader)
    print("Done")
    print("Validating ... ", end='')
    train_metrics = my_trainer.validate(train_loader)
    valid_metrics = my_trainer.validate(valid_loader)
    print("Done")
    print('; '.join([str(x) for x in train_metrics + valid_metrics]))

    if index % 100 == 0:
        my_trainer.make_checkpoint()