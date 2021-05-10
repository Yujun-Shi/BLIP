import torch
import torch.nn as nn
import numpy as np
from .quant_layer import Linear_Q

class Q_MLP(torch.nn.Module):

    def __init__(self, args):
        super(Q_MLP, self).__init__()

        ncha,size,_= args.inputsize
        self.taskcla= args.taskcla
        self.device = args.device

        self.features=torch.nn.Sequential(
            Linear_Q(ncha*size*size, args.ndim, F_prior=args.F_prior, max_bit=args.max_bit),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.classifier.append(nn.Linear(args.ndim,n))

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = torch.nn.functional.relu(self.features(x))
        y=[]
        for t in range(len(self.classifier)):
            y.append(self.classifier[t](x))
        return y

def Net(args):
    return Q_MLP(args)
