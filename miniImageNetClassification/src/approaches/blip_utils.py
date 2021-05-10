import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import sys
sys.path.insert(0, '../')
from networks.quant_layer import Linear_Q, Conv2d_Q

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
def update_Fisher(model):
    #for m in model.features.modules():
    for m in model.modules():
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
            else:
                f_state = batch_conv(m._state, ext_weight, bias=None, \
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

def estimate_fisher(task, device, net, loader):
    net.eval()

    def _save_state(module, input, results):
        module._state = input[0].clone()

    def _save_costate(module, grad_input, grad_output):
        module._costate = grad_output[0].clone()

    # register hooks
    for m in net.modules():
        if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
            m.handle_forward = m.register_forward_hook(_save_state)
            m.handle_backward = m.register_backward_hook(_save_costate)

    criterion = nn.CrossEntropyLoss()

    for i, (images,_) in enumerate(loader):
        images = images.to(device)
        outputs = net(images)[task]

        # sample computation target
        targets = Categorical(logits=outputs).sample()

        loss = criterion(outputs, targets)
        net.zero_grad()
        loss.backward()
        update_Fisher(net)
        net.zero_grad()
        if (i+1) == 100:
            break

    total_data = 1000
    for m in net.modules():
        if isinstance(m, Linear_Q) or isinstance(m, Conv2d_Q):
            m.Fisher_w /= total_data
            if m.bias is not None:
                m.Fisher_b /= total_data
            m.handle_forward.remove()
            m.handle_backward.remove()
