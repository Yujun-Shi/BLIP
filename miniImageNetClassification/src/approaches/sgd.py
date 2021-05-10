import os,sys,time
import numpy as np
import torch
from copy import deepcopy

from .utils import *

class Appr(object):

    def __init__(self,model,args,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000):
        self.model=model

        self.nepochs=args.nepochs
        self.sbatch=args.sbatch
        self.lr=args.lr
        self.momentum = args.momentum
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.checkpoint = args.checkpoint

        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.device = args.device

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr, momentum=self.momentum)

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

        return

    def train_epoch(self,t,loader):
        self.model.train()

        # Loop batches
        for images, targets in loader:
            images,targets = images.to(self.device), targets.to(self.device)

            # Forward
            outputs=self.model.forward(images)
            output=outputs[t]
            loss=self.criterion(output,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
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
                loss=self.criterion(output,targets)
                _,pred=output.max(1)
                hits=(pred==targets).float()

                # Log
                total_loss+=loss.item()*n
                total_acc+=hits.sum().item()
                total_num+=n

        return total_loss/total_num,total_acc/total_num

    def save_model(self,t):
        torch.save({'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(t)))
