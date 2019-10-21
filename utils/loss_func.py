import torch
from torch import nn
import numpy as np
from torch import autograd


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def L1_mask(x, y, mask=None, epsilon=1.001e-5):
    res = torch.abs(x - y)
    b,c,h,w = y.shape
    if mask is not None:
        res = res * mask
        _safe = torch.sum(mask).clamp(epsilon, b*c*h*w+1)
        return torch.sum(res) / _safe
    return torch.mean(res)


def L1_mask_hard_mining(x, y, mask):
    input_size = x.size()
    res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
    with torch.no_grad():
        idx = mask > 0.5
        res_sort = [torch.sort(res[i, idx[i, ...]])[0] for i in range(idx.shape[0])]
        res_sort = [i[int(i.shape[0] * 0.5)].item() for i in res_sort]
        new_mask = mask.clone()
        for i in range(res.shape[0]):
            new_mask[i, ...] = ((mask[i, ...] > 0.5) & (res[i, ...] > res_sort[i])).float()

    res = res * new_mask
    final_res = torch.sum(res) / torch.sum(new_mask)
    return final_res, new_mask
    
def get_gradient(image):
    b, c, h, w = image.shape
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    
    dy = torch.cat([dy, torch.zeros([b, c, 1, w], dtype=dy.dtype).cuda()], axis=2)
    dx = torch.cat([dx, torch.zeros([b, c, h, 1], dtype=dx.dtype).cuda()], axis=3)
    return dx, dy

def L1_grad(pred, gt, mask=None, epsilon=1.001e-5):
    fake_grad_x, fake_grad_y = get_gradient(pred)
    true_grad_x, true_grad_y = get_gradient(gt)

    mag_fake = torch.sqrt(fake_grad_x ** 2 + fake_grad_y ** 2 + epsilon)
    mag_true = torch.sqrt(true_grad_x ** 2 + true_grad_y ** 2 + epsilon)

    return L1_mask(mag_fake, mag_true, mask=mask)

