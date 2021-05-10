from __future__ import print_function
from PIL import Image
import os
import os.path
import sys

import pickle

import torch.utils.data as data
import numpy as np

import torch
from torchvision import transforms


class MiniImageNet(torch.utils.data.Dataset):

    def __init__(self, root, train):
        super(MiniImageNet, self).__init__()
        if train:
            self.name='train'
        else:
            self.name='test'
        with open(os.path.join(root,'{}.pkl'.format(self.name)), 'rb') as f:
            data_dict = pickle.load(f)

        self.data = data_dict['images']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.labels[i]
        return img, label


class iMiniImageNet(MiniImageNet):

    def __init__(self, root, classes, train, transform=None):
        super(iMiniImageNet, self).__init__(root=root, train=train)

        self.transform = transform
        if not isinstance(classes, list):
            classes = [classes]

        self.class_mapping = {c: i for i, c in enumerate(classes)}
        self.class_indices = {}

        for cls in classes:
            self.class_indices[self.class_mapping[cls]] = []

        data = []
        labels = []

        for i in range(len(self.data)):
            if self.labels[i] in classes:
                data.append(self.data[i])
                labels.append(self.class_mapping[self.labels[i]])
                self.class_indices[self.class_mapping[self.labels[i]]].append(i)

        self.data = np.array(data)
        self.labels = labels

    def __getitem__(self, index):

        img, target = self.data[index], self.labels[index]

        if not torch.is_tensor(img):
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, args):
        super(DatasetGen, self).__init__()

        self.seed = args.seed
        self.sbatch=args.sbatch
        self.pc_valid=args.pc_valid
        self.root = args.data_dir

        self.num_tasks = args.ntasks
        self.num_classes = 100

        self.inputsize = [3,84,84]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.train_transformation = transforms.Compose([
                                    transforms.Resize((self.inputsize[1],self.inputsize[2])),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])

        self.test_transformation = transforms.Compose([
                                    transforms.Resize((self.inputsize[1],self.inputsize[2])),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])

        self.taskcla = [[t, int(self.num_classes/self.num_tasks)] for t in range(self.num_tasks)]

        self.indices = {}
        self.dataloaders = {}
        self.idx={}

        self.num_workers = args.workers
        self.pin_memory = True

        np.random.seed(self.seed)
        task_ids = np.split(np.random.permutation(self.num_classes),self.num_tasks)
        self.task_ids = [list(arr) for arr in task_ids]

        self.train_set = {}
        self.train_split = {}
        self.test_set = {}

    def get(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        self.train_set[task_id] = iMiniImageNet(root=self.root, classes=self.task_ids[task_id],
                                                train=True, transform=self.train_transformation)

        self.test_set[task_id] = iMiniImageNet(root=self.root, classes=self.task_ids[task_id],
                                                train=False, transform=self.test_transformation)

        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id], [len(self.train_set[task_id]) - split, split])
        self.train_split[task_id] = train_split

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.sbatch, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory,shuffle=True)
        fisher_loader = torch.utils.data.DataLoader(train_split, batch_size=10, num_workers=1,
                                                   pin_memory=self.pin_memory,shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=self.sbatch,
                                                   num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.sbatch, num_workers=self.num_workers,
                                                  pin_memory=self.pin_memory, shuffle=True)


        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['fisher'] = fisher_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = 'iMiniImageNet-{}-{}'.format(task_id,self.task_ids[task_id])

        print ("Task ID: ", task_id)
        print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Validation set size: {} images of {}x{}".format(len(valid_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Train+Val  set size: {} images of {}x{}".format(len(valid_loader.dataset)+len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        return self.dataloaders
