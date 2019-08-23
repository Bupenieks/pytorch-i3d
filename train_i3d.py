import os, json
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
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
from tqdm import tqdm

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='')
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--ngpu', type=int, default=0)
parser.add_argument('--config', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--save-model', type=str, default='')
parser.add_argument('--sync-bn', type=bool, default=True)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)

world_size = args.ngpu
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,
)

if args.local_rank != 0: tqdm = lambda x, **kwargs : x


def run(num_vids, sync_bn=True, init_lr=0.1, max_steps=64e3, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', val_split='charades/charades.json', batch_size=8*5, save_model='', num_classes=2):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.Tile(64),
                                           videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'training', root, mode, train_transforms)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=args.ngpu,
        rank=args.local_rank,
    )

    def remove_bad_vids(batch):
        batch = list(filter(lambda x:x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
        collate_fn=remove_bad_vids
    )


    val_dataset = Dataset(val_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)    
   

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(2)

    if sync_bn: 
        print("Using SyncBatchNorm")
        i3d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(i3d)

    if args.phase == 'val':
        print("Loading model {}".format(args.save_model))
        i3d.load_state_dict(torch.load(args.save_model))
        
    i3d.cuda()

    i3d = torch.nn.parallel.DistributedDataParallel(
        i3d,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )

    # Send model to its device
    device = torch.device('cuda:{}'.format(args.local_rank))
    i3d = i3d.to(device)

    
    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        if args.local_rank == 0:
            print('Step {}/{}'.format(steps, max_steps))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [args.phase]:#, 'val']: 
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                if args.local_rank != 0: break
                
            tot_acc = 0.0
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            #for data in dataloaders[phase]:
            for it, data in tqdm(enumerate(dataloaders[phase]), total=num_vids//(args.ngpu*batch_size), ncols=100):
                num_iter += 1

                inputs, labels = data
                
                # Send input to device
                inputs, labels = inputs.to(device), labels.to(device)

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
                if num_iter % 10 == 0 and phase == 'val':
                    print('{} Accuracy: {:.4f} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase,tot_acc/num_iter, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter) )

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if args.local_rank == 0 and steps % 10 == 0:
                        print('{} Accuracy: {:.4f} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_acc/(num_steps_per_update), tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        # save model
                        os.system('mkdir -p {}'.format(save_model))
                        torch.save(i3d.module.state_dict(), os.path.join(save_model, str(steps).zfill(6))+'.pt')
                        tot_acc = tot_loss = tot_loc_loss = tot_cls_loss = 0.

            if phase == 'val':
                print('{} Accuracy: {:.4f} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase,tot_acc/num_iter, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter) )
    


if __name__ == '__main__':
    # need to add argparse
    with open(args.config,'r') as fin:
        config = json.load(fin)

    run(**config)

    #run(mode='rgb', root='/data5/ben/gunshots/data/npys', train_split='/data5/ben/gunshots/data/10585train_bin_inp.txt',val_split='/data5/ben/gunshots/data/1500val_inp.txt', batch_size=7)
    #run(mode=config['mode'],
    #    root=config['root'],
    #    train_split=config['train_split'],
    #    val_split=config["val_split"],
    #    batch_size=config['batch_size'],
    #    save_model=args.name)
