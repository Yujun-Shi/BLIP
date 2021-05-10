import numpy as np
import torch
import torch.nn as nn

from .quant_layer import Linear_Q, Conv2d_Q

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNetFeature(torch.nn.Module):

    def __init__(self, args):
        super(AlexNetFeature,self).__init__()

        ncha,size,_ = args.inputsize

        self.mul = args.mul
        self.F_prior = args.F_prior
        self.max_bit = args.max_bit

        self.conv1=Conv2d_Q(ncha,round(64*self.mul), kernel_size=size//8, F_prior=self.F_prior, max_bit=self.max_bit)
        s=compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=Conv2d_Q(round(64*self.mul),round(128*self.mul), kernel_size=size//10, F_prior=self.F_prior, max_bit=self.max_bit)
        s=compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=Conv2d_Q(round(128*self.mul),round(256*self.mul), kernel_size=2, F_prior=self.F_prior, max_bit=self.max_bit)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.2)

        if args.experiment == 'mixture5':
            self.fc1=Linear_Q(round(256*self.mul)*s*s, round(2048*self.mul), F_prior=self.F_prior, max_bit=self.max_bit)
            self.fc2=Linear_Q(round(2048*self.mul), round(2048*self.mul), F_prior=self.F_prior, max_bit=self.max_bit)
        elif args.experiment == 'cifar':
            self.fc1=Linear_Q(round(256*self.mul)*s*s, 2048, F_prior=self.F_prior, max_bit=self.max_bit)
            self.fc2=Linear_Q(2048, 2048, F_prior=self.F_prior, max_bit=self.max_bit)

    def forward(self,x):
        h=self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h=self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h=self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop2(self.relu(self.fc2(h)))
        return h

class AlexNet(nn.Module):

    def __init__(self, args):
        super(AlexNet, self).__init__()
        self.taskcla=args.taskcla

        self.features = AlexNetFeature(args)
        self.last_dim = self.features.fc2.out_features
        self.classifier = nn.ModuleList()
        for t,n in self.taskcla:
            self.classifier.append(nn.Linear(self.last_dim,n))

    def forward(self, x):
        x = self.features(x)

        y = []
        for t in range(len(self.classifier)):
            y.append(self.classifier[t](x))
        return y


def Net(args):
    return AlexNet(args)
