import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_layer import Linear_Q, Conv2d_Q

linear_affine = False

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, F_prior, max_bit):
        super(BasicBlock, self).__init__()
        self.F_prior = F_prior
        self.max_bit = max_bit

        self.conv1 = Conv2d_Q(in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False, F_prior=self.F_prior, max_bit=self.max_bit)
        self.bn1 = nn.BatchNorm2d(planes, affine=linear_affine)
        self.conv2 = Conv2d_Q(planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False, F_prior=self.F_prior, max_bit=self.max_bit)
        self.bn2 = nn.BatchNorm2d(planes, affine=linear_affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_Q(in_planes, self.expansion*planes, kernel_size=1,
                    stride=stride, bias=False, F_prior=self.F_prior, max_bit=self.max_bit),
                nn.BatchNorm2d(self.expansion*planes, affine=linear_affine)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(ResNet, self).__init__()
        self.in_planes = int(64*args.mul)
        self.taskcla = args.taskcla
        self.F_prior = args.F_prior
        self.max_bit = args.max_bit

        ncha,size,_ = args.inputsize

        self.conv1 = Conv2d_Q(ncha, int(64*args.mul), kernel_size=3,
                               stride=2, padding=1, bias=False, F_prior=self.F_prior, max_bit=self.max_bit)
        self.bn1 = nn.BatchNorm2d(int(64*args.mul), affine=linear_affine)
        self.layer1 = self._make_layer(block, int(64*args.mul), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*args.mul), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*args.mul), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*args.mul), num_blocks[3], stride=2)
        self.last_dim = int(args.mul*512*block.expansion)
        self.classifier = nn.ModuleList()
        for t,n in self.taskcla:
            self.classifier.append(nn.Linear(self.last_dim, n))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride, F_prior=self.F_prior, max_bit=self.max_bit))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        y = []
        for linear in self.classifier:
            y.append(linear(out))
        return y

def Net(args):
    return ResNet(BasicBlock, [2, 2, 2, 2], args)
