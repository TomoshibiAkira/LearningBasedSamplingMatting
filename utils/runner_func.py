import numpy as np
import torch
import torch.nn.functional as F

class LRDecay(object):
    def stair(self, iter):
        new_lr = self.lr * self.schedule[self.current_schedule][1]
        if self.current_schedule == len(self.schedule)-1:
            pass
        elif iter == self.schedule[self.current_schedule+1][0] - 1:
            self.current_schedule += 1
            print ('Next iter LR changed to', self.lr * self.schedule[self.current_schedule][1])
        return new_lr
        
    def power(self, iter):
        # TODO:
        if self.current_schedule == len(self.schedule)-1:
            new_lr = self.lr * self.schedule[self.current_schedule-1][1]
        else:
            steps = self.schedule[self.current_schedule][0] - \
                    self.schedule[self.current_schedule+1][0]
            ratio = self.schedule[self.current_schedule][1] - \
                    self.schedule[self.current_schedule+1][1]
            cur = float(iter - self.schedule[self.current_schedule][0]) / float(steps) * ratio
            cur_ratio = self.schedule[self.current_schedule][1] - cur
            new_lr = self.lr * cur_ratio
            if iter == self.schedule[self.current_schedule+1][0] - 1:
                self.current_schedule += 1
                print ('Change to next schedule.')
        return new_lr            
        
    def __init__(self, mode, lr, schedule=[], power_p=0.9):
        # scheule: list of tuple, tuple[0] is iter, tuple[1] is ratio
        self.lr = lr
        schedule = list(schedule)
        schedule.append((0, 1.0))
        self.schedule = sorted(schedule)
        if mode == 'stair':
            self.func = self.stair
        elif mode == 'power':
            self.func = self.power
            self.power_p = power_p
        self.current_schedule = 0
        
    def __call__(self, iter):
        return self.func(iter)

def get_all_params(model):
    for i in model.modules():
        for j in i.parameters():
            if j.requires_grad:
                yield j
                
def get_joint_params(models):
    assert isinstance(models, list) or isinstance(models, tuple)
    for m in models:
        if m.training:
            for i in m.modules():
                for j in i.parameters():
                    if j.requires_grad:
                        yield j

# helpful funcs in training process
def print_loss_dict(loss):
    for key, value in loss.items():
        print(key, ': ', value)


def write_loss_dict(loss, writer, iter, prefix='loss_'):
    for key, value in loss.items():
        writer.add_scalar(prefix+key, value, iter)
        
def write_image_summary(tag, image, writer, iter, max_write=4, cvt_to_rgb=True):
    #imgs = image.split(1, dim=0)
    cvt_to_rgb = cvt_to_rgb and (image.shape[1] == 3)
    for i in range(min(image.shape[0], max_write)):
        img = torch.flip(image[i], dims=[0]) if cvt_to_rgb else image[i]
        writer.add_image(tag+'/{}'.format(i), img, iter)


def adjust_learning_rate(optimizer, iter, decay):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #decay = decay_ratio**(sum(iter >= np.array(iter_step)) - sum((iter-1) >= np.array(iter_step)))
    _lr = decay(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = _lr
    return _lr


def down_sample(x, scalor=2, mode='bilinear'):
    if mode == 'bilinear':
        x = F.avg_pool2d(x, kernel_size=scalor, stride=scalor)
    elif mode == 'nearest':
        x = F.max_pool2d(x, kernel_size=scalor, stride=scalor)

    return x
    
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def watch_gradient(model, writer, name, n_iter):
    for param_name, param in model.named_parameters():
        if name in param_name:
            if param.grad is not None:
                writer.add_histogram(param_name, param.grad.data, n_iter)
