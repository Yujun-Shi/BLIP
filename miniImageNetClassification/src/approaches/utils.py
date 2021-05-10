import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def fisher_matrix_diag(t,trainloader,model,sbatch=20):
    # Init
    criterion = torch.nn.CrossEntropyLoss()
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    for images,targets in tqdm(trainloader,desc='Fisher diagonal',ncols=100,ascii=True):
        images,targets=images.cuda(),targets.cuda()
        # Forward and backward
        model.zero_grad()
        outputs=model.forward(images)
        loss=criterion(outputs[t],targets)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n,_ in model.named_parameters():
            fisher[n]=fisher[n]/len(trainloader)
    return fisher

def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs, dim=1)
    tar=torch.nn.functional.softmax(targets, dim=1)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce
