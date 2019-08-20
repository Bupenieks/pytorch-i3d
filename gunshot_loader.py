import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import random
import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2


def read_npy_file(fname): 
    data = np.load(fname)
    return data.astype(np.float32)


def make_dataset(labels_file, split, root):
    class_file = '/data5/ben/gunshots/classes.txt'
    with open(class_file, 'r') as f:
        classes = [c.strip() for c in f]


    with open(labels_file, 'r') as f:
        file_list = [line.strip().split(',') for line in f]
        random.shuffle(file_list)

        f_list = []
        label_list=[]
        for fname, label in file_list:  
            f_list.append(os.path.join(root, fname))
            label_list.append(classes.index(label))
    
    num_data_points = len(f_list)

    dataset = list(zip(f_list, label_list))

    return dataset




class VideoLoader(data_utl.Dataset):
    def __init__(self, split_file, split, root, mode, transforms=None, save_dir='', num=0):
        
        self.data = make_dataset(split_file, split, root)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        try:
            imgs, label = self.data[index]
            imgs = read_npy_file(imgs)
            imgs = self.transforms(imgs)
            imgs = imgs.transpose([3,0,1,2])
        
            label_vec = np.zeros((2, 64), np.float32)
            label_vec[label,:] = 1


            return torch.from_numpy(imgs), torch.from_numpy(label_vec)
        except ValueError as e:
            return None

    def __len__(self):
        return len(self.data)
