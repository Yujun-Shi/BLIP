import os,sys,time
import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '../')
from networks.quant_layer import Linear_Q, Conv2d_Q


class Appr(object):

    def __init__(self,model,args,lr_patience=5,lr_factor=3,lr_min=1e-4):
        self.model=model
        self.device = args.device

        self.init_lr=args.lr
        self.momentum=args.momentum
        self.weight_decay=args.weight_decay

        # patience decay parameters
        self.lr_patience = lr_patience
        self.lr_factor=lr_factor
        self.lr_min=lr_min

        self.sbatch=args.sbatch
        self.nepochs=args.nepochs

        self.checkpoint = args.checkpoint
        self.experiment=args.experiment
        self.num_tasks=args.num_tasks

        self.criterion = nn.CrossEntropyLoss()

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.init_lr
        return torch.optim.SGD(self.model.parameters(),lr=lr,momentum=self.momentum, weight_decay=self.weight_decay)

    def train(self,t,trainloader,valloader):
        self.optimizer = self._get_optimizer()

        best_model = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf
        patience = self.lr_patience
        lr=self.init_lr

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                self.train_epoch(t,trainloader)
                train_loss,train_acc=self.eval(t,trainloader)

                print('| Epoch {:3d}| Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,valloader)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

                if math.isnan(valid_loss) or math.isnan(train_loss):
                    print("saved best model and quit because loss became nan")
                    break
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    patience=self.lr_patience
                    best_model=copy.deepcopy(self.model.state_dict())
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
        except KeyboardInterrupt:
            print()

        # Restore best
        self.model.load_state_dict(copy.deepcopy(best_model))

    def train_epoch(self, t, loader):
        self.model.train()

        # freeze bn after the first task
        if t > 0:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        for images,targets in loader:
            images, targets = images.to(self.device), targets.to(self.device)

            # Forward
            outputs=self.model(images)
            outputs_t=outputs[t]
            loss = self.criterion(outputs_t, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # Update parameters
            self.optimizer.step()

            for m in self.model.modules():
                if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
                    m.clipping()
        return

    def eval(self, t, loader):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        with torch.no_grad():
            for images, targets in loader:
                n = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)

                # Forward
                outputs=self.model(images)
                outputs_t=outputs[t]
                loss = self.criterion(outputs_t, targets)

                _,pred=outputs_t.max(1, keepdim=True)

                total_loss += loss.item()*n
                total_acc += pred.eq(targets.view_as(pred)).sum().item() 
                total_num += n

        return total_loss/total_num, total_acc/total_num

    def save_model(self,t):
        torch.save({'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(t)))
