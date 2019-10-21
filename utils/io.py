import torch
import torch.nn as nn


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, optimizers=None, strict=False):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=strict)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']

def save_ckpt_epoch(ckpt_name, models, optimizers, n_epoch):
    ckpt_dict = {'n_epoch': n_epoch}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)

def load_ckpt_epoch(ckpt_name, models, optimizers=None, strict=False, conv1_chan=None):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        if prefix == 'model' and conv1_chan:
            trans_state_dict = {}
            for i in ckpt_dict[prefix].keys():
                if 'module.encoder.conv1.weight' == i:
                    conv1_weight = ckpt_dict[prefix][i]
                    _add = conv1_chan - conv1_weight.shape[1]
                    if _add != 0:
                        conv1_add = torch.zeros([conv1_weight.shape[0], _add,
                                                 conv1_weight.shape[2],
                                                 conv1_weight.shape[3]],
                                                 dtype=conv1_weight.dtype)
                        weight = torch.cat([conv1_weight, conv1_add], axis=1)
                    else:
                        weight = conv1_weight
                else:
                    weight = ckpt_dict[prefix][i]
                trans_state_dict[i] = weight
            ckpt_dict[prefix] = trans_state_dict
        model.load_state_dict(ckpt_dict[prefix], strict=strict)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_epoch'] # TODO 'n_epoch'


def load_corr_ckpt(ckpt_name, model):
    ckpt_dict = torch.load(ckpt_name)
    model.load_state_dict(ckpt_dict, strict=False)
