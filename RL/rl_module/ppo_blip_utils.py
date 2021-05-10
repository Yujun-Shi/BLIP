import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .quant_layer import Linear_Q, Conv2d_Q

__all__ = ['update_fisher_exact']

def batch_conv(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, b_j, c, h, w = x.shape
    b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w)
    weight = weight.view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

    out = F.conv2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                    padding=padding)
    out = out.view(b_j, b_i, out_channels, out.shape[-2], out.shape[-1])
    out = out.permute([1, 0, 2, 3, 4])

    if bias is not None:
        out = out + bias.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

    return out

# use batch convolution here to improve efficiency
def update_fisher_exact(model):
    for m in model.features.modules():
        if isinstance(m, Conv2d_Q):
            # conv layer
            # smart batch conv version
            # extend state to (N,1,C,H,W)
            N = m._state.shape[0]
            m._state = m._state.unsqueeze(1)

            # quantize weight
            # sync_weight is already called
            weight_q = m.weight.detach()

            # extend weight to (N, C_out, C_in, K_h, K_w)
            # multiply by the scaling factor here
            ext_weight = weight_q.unsqueeze(0).repeat(N,1,1,1,1)
            ext_weight.requires_grad_(True)
            if m.bias is not None:
                # extend bias to (N, C_out)
                ext_bias = m.bias.detach().unsqueeze(0).repeat(N,1)
                ext_bias.requires_grad_(True)

            f_state = batch_conv(m._state, ext_weight, bias=ext_bias, \
                stride=m.stride, padding=m.padding, dilation=m.dilation)
            H = (f_state.squeeze(1)*m._costate).sum()

            if m.bias is not None:
                ext_grad_w, ext_grad_b = torch.autograd.grad(H, (ext_weight, ext_bias))
            else:
                ext_grad_w = torch.autograd.grad(H, ext_weight)[0]

            with torch.no_grad():
                # record fisher information
                m.Fisher_w.data.add_((ext_grad_w.pow_(2)).sum(dim=0).data)
                if m.bias is not None:
                    m.Fisher_b.data.add_((ext_grad_b.pow_(2)).sum(dim=0).data)

        elif isinstance(m, Linear_Q):
            # extend state to (N, 1, C_in)
            N = m._state.shape[0]
            m._state = m._state.unsqueeze(1)

            # quantize weight
            # sync_weight is already called
            weight_q = m.weight.detach()

            # extend weight to (N, C_out, C_in)
            ext_weight = weight_q.unsqueeze(0).repeat(N,1,1)
            ext_weight.requires_grad_(True)
            ext_weight = ext_weight.permute(0,2,1)  # (N, C_in, C_out)

            f_state = torch.bmm(m._state, ext_weight).squeeze(1)  # (N, C_out)
            if m.bias is not None:
                bias_q = m.bias.detach()
                ext_bias = bias_q.unsqueeze(0).repeat(N,1).requires_grad_(True) # (N, C_out)
                f_state += ext_bias
            H = (f_state*m._costate).sum()

            if m.bias is not None:
                ext_grad_w,ext_grad_b = torch.autograd.grad(H, (ext_weight,ext_bias))
            else:
                ext_grad_w = torch.autograd.grad(H, ext_weight)[0]

            ext_grad_w = ext_grad_w.permute(0,2,1)  # (N, C_out, C_in)

            with torch.no_grad():
                # record fisher information
                m.Fisher_w.data.add_((ext_grad_w.pow_(2)).sum(dim=0).data)
                if m.bias is not None:
                    m.Fisher_b.data.add_((ext_grad_b.pow_(2)).sum(dim=0).data)
