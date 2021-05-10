import os,sys,time
import numpy as np
import torch
from copy import deepcopy

from .utils import *

import sys
sys.path.insert(0, '../')

class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self,model,args,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=100,lamb=5000):
        self.model=model
        self.model_old=None
        self.fisher=None

        self.nepochs=args.nepochs
        self.sbatch=args.sbatch
        self.lr=args.lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.checkpoint = args.checkpoint

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.lamb=lamb

        self.device = args.device
        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train(self,t,trainloader,valloader):
        best_loss=np.inf
        best_model=deepcopy(self.model.state_dict())
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            self.train_epoch(t,trainloader)
            train_loss,train_acc=self.eval(t,trainloader)
            print('| Epoch {:3d}| Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,valloader)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=deepcopy(self.model.state_dict())
                patience=self.lr_patience
                print(' *',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=self.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<self.lr_min:
                        print()
                        break
                    patience=self.lr_patience
                    self.optimizer=self._get_optimizer(lr)
            print()

        # Restore best
        self.model.load_state_dict(best_model)

        # Update old
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        freeze_model(self.model_old) # Freeze the weights

        # Fisher ops
        if t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=fisher_matrix_diag(t,trainloader,self.model)
        if t>0:
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)

        return

    def train_epoch(self,t,loader):
        self.model.train()

        # Loop batches
        for images, targets in loader:
            images,targets = images.to(self.device), targets.to(self.device)

            # Forward current model
            outputs=self.model.forward(images)
            output=outputs[t]
            loss=self.criterion(t,output,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,loader):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        # Loop batches
        with torch.no_grad():
            for images,targets in loader:
                n = images.shape[0]
                images,targets = images.to(self.device),targets.to(self.device)

                # Forward
                outputs=self.model.forward(images)
                output=outputs[t]
                loss=self.criterion(t,output,targets)
                _,pred=output.max(1)
                hits=(pred==targets).float()

                # Log
                total_loss+=loss.item()*n
                total_acc+=hits.sum().item()
                total_num+=n

        return total_loss/total_num,total_acc/total_num

    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

        return self.ce(output,targets)+self.lamb*loss_reg

    def save_model(self,t):
        torch.save({'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(t)))
