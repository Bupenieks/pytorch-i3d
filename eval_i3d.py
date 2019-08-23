import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

from gunshot_loader import VideoLoader as Dataset

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='')
args = parser.parse_args()


def run(num_vids, init_lr=0.1, max_steps=64e3, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', val_split='charades/charades.json', batch_size=8*5, save_model='', num_classes=2):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    val_dataset = Dataset(val_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)    

    dataloaders = {'train': None, 'val': val_dataloader}
    datasets = {'train': None, 'val': val_dataset}
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    i3d.replace_logits(2)
    i3d.load_state_dict(torch.load(save_model))

    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_acc = 0.0
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                print(num_iter)
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                logits, _ = torch.max(per_frame_logits, dim=2)
                labels, _ =  torch.max(labels, dim=2)

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
                tot_cls_loss += cls_loss.item()

                loss = (cls_loss)/num_steps_per_update
                tot_loss += loss.item()
                loss.backward()

                predictions = torch.nn.Softmax(dim=-1)(logits)

                bin_predictions = predictions >= 0.5
                
                acc = (bin_predictions * labels.byte()).float().sum() / batch_size
                tot_acc += acc

                if num_iter % 1 == 0:
                    print("{}/1500".format(num_iter))

            if phase == 'val':
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))
    


if __name__ == '__main__':
    # need to add argparse
    with open(args.config,'r') as fin:
        config = json.load(fin)

    run(**config)
