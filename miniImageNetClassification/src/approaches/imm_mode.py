import os,sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
from collections import defaultdict

from .utils import *

class Appr(object):
    """ Class implementing the Incremental Moment Matching (mode) approach described in https://arxiv.org/abs/1703.08475 """

    def __init__(self,model,args,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=1000,lamb=0.01):
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

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.lamb=lamb      # Grid search = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]; best was 1
        #self.alpha=0.5     # We assume the same alpha for all tasks. Unrealistic to tune one alpha per task (num_task-1 alphas) when we have a lot of tasks.

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
            train_loss,train_acc = self.eval(t,trainloader)
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

        # Model update
        if t==0:
            self.fisher=fisher_matrix_diag(t,trainloader,self.model)
        else:
            fisher_new=fisher_matrix_diag(t,trainloader,self.model)
            for (n,p),(_,p_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                p=fisher_new[n]*p+self.fisher[n]*p_old
                self.fisher[n]+=fisher_new[n]
                p/=(self.fisher[n]==0).float()+self.fisher[n]

        # Old model save
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        freeze_model(self.model_old)

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

        # L2 multiplier loss
        loss_reg=0
        if t>0:
            for p,p_old in zip(self.model.parameters(),self.model_old.parameters()):
                loss_reg+=(p-p_old).pow(2).sum()/2

        # Cross entropy loss
        loss_ce=self.ce(output,targets)

        return loss_ce+self.lamb*loss_reg

    def save_model(self,t):
        torch.save({'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(t)))
