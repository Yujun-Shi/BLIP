import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

def uniform_quantize(bit_alloc, upper_bound):
    class qfn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            # first, normalize to (-1, 1)
            input = input / upper_bound

            # then, perform quantization
            n_t = (2**bit_alloc).float()
            out = torch.round(input * n_t) / n_t
            out[bit_alloc==0] = 0.0  # quantize a number to ``0 bit'' is basically set it to 0

            # Finally, scale back to original scale
            out = out * upper_bound
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return qfn().apply

# smaller prior means more bits for the first task
class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, max_bit=20, F_prior=1e-18):
        super(Linear_Q, self).__init__(in_features, out_features, bias)
        self.max_bit = max_bit

        self.bound = 6.0*math.sqrt(1/in_features)
        # dealing each layer's prior separately
        self.F_prior = F_prior*in_features

        self.register_buffer('prev_weight', torch.zeros_like(self.weight))
        self.register_buffer('bit_alloc_w', torch.zeros_like(self.weight, dtype=torch.int64))
        self.register_buffer('Fisher_w', torch.zeros_like(self.weight))
        self.register_buffer('Fisher_w_old', self.F_prior*torch.ones_like(self.weight))
        self.weight.data.normal_(0.0, math.sqrt(1/in_features))
        self.weight.data.clamp_(-self.bound, self.bound)
        if self.bias is not None:
            self.register_buffer('prev_bias', torch.zeros_like(self.bias))
            self.register_buffer('bit_alloc_b', torch.zeros_like(self.bias, dtype=torch.int64))
            self.register_buffer('Fisher_b', torch.zeros_like(self.bias))
            self.register_buffer('Fisher_b_old', self.F_prior*torch.ones_like(self.bias))
            self.bias.data.zero_()

    # synchronize continuous weight with quantized weight
    def sync_weight(self):
        with torch.no_grad():
            weight_q = uniform_quantize(self.bit_alloc_w, self.bound)(self.weight)
            self.weight.data.copy_(weight_q.data)
            self.prev_weight.data.copy_(weight_q.data)

            if self.bias is not None:
                bias_q = uniform_quantize(self.bit_alloc_b, 1.0)(self.bias)
                self.bias.data.copy_(bias_q.data)
                self.prev_bias.data.copy_(bias_q.data)

    # update bits according to info gain
    # C is a hyper parameter on scale translate between info gain and actual bit length
    # default is 0.5 according to our derivation
    def update_bits(self, task, C=0.5/math.log(2)):
        with torch.no_grad():
            info_gain_w = C*torch.log((self.Fisher_w + self.Fisher_w_old*(task+1))/(self.Fisher_w_old*(task+2)))
            bit_grow_w = torch.ceil(info_gain_w).long()
            bit_grow_w.clamp_(min=0)
            self.bit_alloc_w.add_(bit_grow_w)
            self.bit_alloc_w.clamp_(max=self.max_bit)

            if self.bias is not None:
                info_gain_b = C*torch.log((self.Fisher_b + self.Fisher_b_old*(task+1))/(self.Fisher_b_old*(task+2)))
                bit_grow_b = torch.ceil(info_gain_b).long()
                bit_grow_b.clamp_(min=0)
                self.bit_alloc_b.add_(bit_grow_b)
                self.bit_alloc_b.clamp_(max=self.max_bit)

    # clipping the parameters to the accepted range
    # should be called after every optimizer.step()
    def clipping(self):
        with torch.no_grad():
            # clippping to the effective range centered around prev_weight
            weight_diff = self.weight.data - self.prev_weight.data
            range_data = self.bound*(0.5**self.bit_alloc_w.float())
            clipped_weight_diff = torch.max(torch.min(weight_diff, range_data), -range_data)
            self.weight.data.copy_(self.prev_weight.data + clipped_weight_diff.data)

            # clipped the above weight to effective range
            range_data = self.bound*(1.0 - 0.5**self.max_bit)
            self.weight.data.clamp_(-range_data, range_data)

            if self.bias is not None:
                # clippping to the effective range centered around prev_bias
                bias_diff = self.bias.data - self.prev_bias.data
                range_data = 0.5**self.bit_alloc_b.float()
                clipped_bias_diff = torch.max(torch.min(bias_diff, range_data), -range_data)
                self.bias.data.copy_(self.prev_bias.data + clipped_bias_diff.data)

                # clipped the above bias to effective range
                range_data = 1.0 - 0.5**self.max_bit
                self.bias.data.clamp_(-range_data, range_data)

    def update_fisher(self, task):
        self.Fisher_w_old.data.mul_(task+1).add_(self.Fisher_w.data).div_(task+2)
        self.Fisher_w.zero_()
        if self.bias is not None:
            self.Fisher_b_old.data.mul_(task+1).add_(self.Fisher_b.data).div_(task+2)
            self.Fisher_b.zero_()

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=1, groups=1, bias=True, max_bit=20, F_prior=1e-18):
        super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias)
        self.max_bit = max_bit
        self.bound = 6.0*math.sqrt(1.0/(in_channels*kernel_size*kernel_size))
        # dealing each layer's prior separately
        self.F_prior = F_prior*in_channels*kernel_size*kernel_size

        self.register_buffer('prev_weight', torch.zeros_like(self.weight))
        self.register_buffer('bit_alloc_w', torch.zeros_like(self.weight, dtype=torch.int64))
        self.register_buffer('Fisher_w', torch.zeros_like(self.weight))
        self.register_buffer('Fisher_w_old', self.F_prior*torch.ones_like(self.weight))
        self.weight.data.normal_(0.0, math.sqrt(1.0/(in_channels*kernel_size*kernel_size)))
        self.weight.data.clamp_(-self.bound, self.bound)
        if self.bias is not None:
            self.register_buffer('prev_bias', torch.zeros_like(self.bias))
            self.register_buffer('bit_alloc_b', torch.zeros_like(self.bias, dtype=torch.int64))
            self.register_buffer('Fisher_b', torch.zeros_like(self.bias))
            self.register_buffer('Fisher_b_old', self.F_prior*torch.ones_like(self.bias))
            self.bias.data.zero_()

    # synchronize continuous weight with quantized weight
    def sync_weight(self):
        with torch.no_grad():
            weight_q = uniform_quantize(self.bit_alloc_w, self.bound)(self.weight)
            self.weight.data.copy_(weight_q.data)
            self.prev_weight.data.copy_(weight_q.data)

            if self.bias is not None:
                bias_q = uniform_quantize(self.bit_alloc_b, 1.0)(self.bias)
                self.bias.data.copy_(bias_q.data)
                self.prev_bias.data.copy_(bias_q.data)

    # update bits according to info gain
    # C is a hyper parameter on scale translate between info gain and actual bit length
    # default is 0.5 according to our derivation
    def update_bits(self, task, C=0.5/math.log(2)):
        with torch.no_grad():
            info_gain_w = C*torch.log((self.Fisher_w + self.Fisher_w_old*(task+1))/(self.Fisher_w_old*(task+2)))
            bit_grow_w = torch.ceil(info_gain_w).long()
            bit_grow_w.clamp_(min=0)
            self.bit_alloc_w.add_(bit_grow_w)
            self.bit_alloc_w.clamp_(max=self.max_bit)

            if self.bias is not None:
                info_gain_b = C*torch.log((self.Fisher_b + self.Fisher_b_old*(task+1))/(self.Fisher_b_old*(task+2)))
                bit_grow_b = torch.ceil(info_gain_b).long()
                bit_grow_b.clamp_(min=0)
                self.bit_alloc_b.add_(bit_grow_b)
                self.bit_alloc_b.clamp_(max=self.max_bit)

    # clipping the parameters to the accepted range
    # should be called after every optimizer.step()
    def clipping(self):
        with torch.no_grad():
            # clippping to the effective range centered around prev_weight
            weight_diff = self.weight.data - self.prev_weight.data
            range_data = self.bound*(0.5**self.bit_alloc_w.float())
            clipped_weight_diff = torch.max(torch.min(weight_diff, range_data), -range_data)
            self.weight.data.copy_(self.prev_weight.data + clipped_weight_diff.data)

            # clipped the above weight to effective range
            range_data = self.bound*(1.0 - 0.5**self.max_bit)
            self.weight.data.clamp_(-range_data, range_data)

            if self.bias is not None:
                # clippping to the effective range centered around prev_bias
                bias_diff = self.bias.data - self.prev_bias.data
                range_data = 0.5**self.bit_alloc_b.float()
                clipped_bias_diff = torch.max(torch.min(bias_diff, range_data), -range_data)
                self.bias.data.copy_(self.prev_bias.data + clipped_bias_diff.data)

                # clipped the above bias to effective range
                range_data = 1.0 - 0.5**self.max_bit
                self.bias.data.clamp_(-range_data, range_data)

    def update_fisher(self, task):
        self.Fisher_w_old.data.mul_(task+1).add_(self.Fisher_w.data).div_(task+2)
        self.Fisher_w.zero_()
        if self.bias is not None:
            self.Fisher_b_old.data.mul_(task+1).add_(self.Fisher_b.data).div_(task+2)
            self.Fisher_b.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)

if __name__ == '__main__':
    # debugging code
    conv = Conv2d_Q(3,3,kernel_size=4)
    conv.bit_alloc_w[0][0][0][0] = 10
    conv.sync_weight()
