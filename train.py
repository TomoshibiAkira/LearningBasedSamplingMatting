import sys, os
import argparse
import collections
import yaml
import datetime
import multiprocessing as mp
import torch
import numpy as np
import cv2
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils.loss_func as L
from dataset.DeepMattingTrain import DIMHeavyComposeSet
from models.alphagan_gen import \
    AlphaGANGenerator, ENCODER_DICT, DECODER_DICT
from models import DeepFill
from utils.io import *
from utils.runner_func import *

BASE_SEED = 7777777    

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use. Default to use all available ones')
    parser.add_argument('--data', help='Path to LMDB dataset')
    parser.add_argument('--init_pretrain', help='use a pretrained model as initialization')
    parser.add_argument('--init_ckpt', help='load a checkpoint as initialization')
    parser.add_argument('--resume', help='resume from a checkpoint with optimizer parameter attached')
    parser.add_argument('--n_threads', type=int, default=16, help='number of workers for dataflow.')
    parser.add_argument('--sync_bn', action='store_true', help='use SyncBatchNorm')
    parser.add_argument('--lr', type=float, default=1e-5, help='initial lr for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument('--batch', default=16, type=int, help="Batch size in total.")
    parser.add_argument('-o', '--optimizer', help='Optimizer used in training.',
                        default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--exp_name', help='saving model name')
    parser.add_argument('-d', '--decoder', default='alphagan-decoder', choices=DECODER_DICT.keys())
    parser.add_argument('-e', '--encoder', default='resnet50-aspp', choices=ENCODER_DICT.keys())
    parser.add_argument('-s', '--size', default=None, help='the size of the input image, use when the network doesn\'t \
                                                            support arbitrary size, split with \'x\'')
    parser.add_argument('--freeze_bn', action='store_true', help='Whether freeze bn while training.')
    parser.add_argument('--freeze_dropout', action='store_true', help='Whether freeze dropout while training.')
    
    parser.add_argument('--max_epoch', type=int, default=80, help='Maximum training epoch')
    parser.add_argument('--image_every', type=int, default=100, help='Write image summary every X step during training')
    parser.add_argument('--lr_ratio', nargs='+', type=float, default=[], help='lr ratio schedule')
    parser.add_argument('--lr_schedule', nargs='+', type=int, default=[], help='lr decay schedule')
    parser.add_argument('--lr_decay_strat', choices=['stair', 'power'], default='stair', help='lr decay strategy')
    parser.add_argument('--lr_decay_power', default=0.9, type=float, help='power parameter in power decay strategy')
    parser.add_argument('--bg_model_weight', default='BGDFV1.pth')
    parser.add_argument('--fg_model_weight', default='FGDFV1.pth')
    parser.add_argument('--write_graph', action='store_true', help='Write network graph to tensorboardX. Need some extra GPU.')
    args = parser.parse_args()
    return args
    
def write_image(out, writer, step, prefix=''):
    with torch.no_grad():
        if step % args.image_every == 0:
            fg_gt, bg_gt, gt, tri, fg_, bg_, rgb_, alpha, comp, valid_mask = out[:-1]
            f = torch.where(tri == 0, torch.zeros_like(fg_), torch.where(valid_mask == 1, fg_, rgb_))
            b = torch.where(tri == 2, torch.ones_like(bg_), torch.where(valid_mask == 1, bg_, rgb_))
            fg = torch.where(tri == 0, torch.zeros_like(fg_), (fg_gt + 1.) / 2.0)
            bg = torch.where(tri == 2, torch.ones_like(bg_), (bg_gt + 1.) / 2.0)
            write_image_summary(prefix+'pred/alpha', alpha, writer, step)
            write_image_summary(prefix+'pred/comp', comp, writer, step)
            write_image_summary(prefix+'pred/fg', f, writer, step)
            write_image_summary(prefix+'pred/bg', b, writer, step)
            write_image_summary(prefix+'input/valid_mask', valid_mask, writer, step)
            write_image_summary(prefix+'input/rgb', rgb_, writer, step)
            write_image_summary(prefix+'input/fg', fg, writer, step)
            write_image_summary(prefix+'input/bg', bg, writer, step)
            write_image_summary(prefix+'input/gt', gt, writer, step)
            write_image_summary(prefix+'input/trimap', tri.float() / 2.0, writer, step)

def forward(net, BG, FG, dp, loss):
    rgb, fg_gt, bg_gt, gt, tri, small_tri = dp
    with torch.no_grad():
        # rgb, fg_gt, bg_gt is in [-1, 1]
        # gt is in [0, 1]
        # tri is 1-channel trimap in {0, 1, 2}, 0 BG 1 UN 2 FG
        # preprocess which does not need gradients
        rgb = rgb.cuda().float().clamp(-1., 1.) # [b, 3, h, w]
        tri = tri.cuda().float().clamp(0., 2.)  # [b, 1, h, w]
        gt = gt.cuda().float().clamp(0., 1.)    # [b, 1, h, w]
        fg_gt = fg_gt.cuda().float().clamp(-1., 1.) # [b, 3, h, w]
        bg_gt = bg_gt.cuda().float().clamp(-1., 1.) # [b, 3, h, w]
        small_tri = small_tri.cuda().float().clamp(0., 2.)
        mask = torch.eq(tri, 1.)
        
        # bg and fg should be a float32 tensor in [0, 1]
        f_u_mask = (tri>0.01).float()
        b_u_mask = (tri<1.99).float()
        small_fumask = (small_tri>0.01).float()
        small_bumask = (small_tri<1.99).float()
        # bg_pred and fg_pred are already in [-1, 1]
        _, bg_pred, _ = BG(rgb, f_u_mask, small_fumask)
        bg_pred_ = torch.where(mask, bg_pred, rgb) #mask * bg_pred + (1 - mask) * rgb
        _, fg_pred, _ = FG(rgb, b_u_mask, small_bumask, bg_img=bg_pred_)
        fg_ = ((fg_pred + 1.0) / 2.0).clamp(0., 1.)
        bg_ = ((bg_pred + 1.0) / 2.0).clamp(0., 1.)
        rgb_ = ((rgb + 1.0) / 2.0).clamp(0., 1.)
        
        input_x = torch.cat([rgb, fg_pred, bg_pred, tri-1.], axis=1)
                
    # network forward
    pred = net(input_x)
    alpha = torch.where(mask, pred, tri / 2.0)
    # composition
    comp = fg_ * alpha + bg_ * (1. - alpha)
    
    # loss calculation
    valid_mask = mask.float()
    loss['L_alpha'] = L.L1_mask(alpha, gt, valid_mask)
    loss['L_comp'] = L.L1_mask(comp, rgb_, valid_mask)
    loss['L_grad'] = L.L1_grad(alpha, gt, valid_mask)
    loss['L_total'] = (loss['L_alpha'] + loss['L_comp']) * 0.5 + loss['L_grad']
    
    return [fg_gt, bg_gt, gt, tri, fg_, bg_, rgb_, alpha, comp, valid_mask, loss]

def main(args):
    ########################## set cuda environment
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        if len(args.gpu.split(',')) == 1:
            args.sync_bn = False            
    
    ########################## set output log dir, write config yml, set summary writer
    if args.exp_name is None:
        exp_name = '{}-{}-b{}-{}-{}'.format(
            args.encoder, args.decoder, args.batch, 
            os.path.dirname(args.data) if args.data.endswith('/') else os.path.basename(args.data),
            datetime.datetime.now().strftime("%y%m%d-%H%M")
        )
    else:
        exp_name = args.exp_name

    model_save_dir = './train_log/'+exp_name
    log_dir = model_save_dir

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
        yaml.dump(vars(args), f)

    writer = SummaryWriter(log_dir=log_dir)

    ########################## random seed initialization
    torch.manual_seed(BASE_SEED)
    # ONLY USE THESE TWO LINES FOR DEBUGGING
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    ########################## init net, deal with sync_bn
    net = AlphaGANGenerator(input=10,
                            encoder=args.encoder,
                            decoder=args.decoder,
                            freeze_bn=args.freeze_bn,
                            freeze_dropout=args.freeze_dropout)
    if args.sync_bn:
        mp.set_start_method('forkserver')
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        torch.distributed.init_process_group(backend="nccl")
        dataparallel_func = nn.parallel.DistributedDataParallel
    else:
        dataparallel_func = nn.DataParallel
    net = dataparallel_func(net).cuda()
    net.train()
    
    ########################## build FG / BG and load weight for them
    BG_sample_net = DeepFill.Generator(build_refine=True)
    FG_sample_net = DeepFill.Generator(build_refine=True, is_fg=True)
    BG_sample_net = nn.DataParallel(BG_sample_net).cuda()
    FG_sample_net = nn.DataParallel(FG_sample_net).cuda()
    BG_sample_net.eval()
    FG_sample_net.eval()
    load_ckpt_epoch(args.bg_model_weight, [('model', BG_sample_net)], strict=True)
    print ('BG sampling net loaded weight from '+args.bg_model_weight)
    load_ckpt_epoch(args.fg_model_weight, [('model', FG_sample_net)], strict=True)
    print ('FG sampling net loaded weight from '+args.fg_model_weight)
    
    ########################## setting up optimizer and lr decay, write graph
    if args.write_graph:
        writer.add_graph(net.module, \
                        (torch.zeros([1, 3, size[0], size[1]]).cuda(),  # rgb
                         torch.zeros([1, 3, size[0], size[1]]).cuda(),  # fg
                         torch.zeros([1, 3, size[0], size[1]]).cuda(),  # bg
                         torch.zeros([1, 1, size[0], size[1]]).cuda(),))# tri
                         
    if args.optimizer == 'adam':
        optimizer = optim.Adam([{'params': get_all_params(net.module), 'lr': args.lr}],
                            lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW([{'params': get_all_params(net.module), 'lr': args.lr}],
                            lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD([{'params': get_all_params(net.module), 'lr': args.lr}],
                            lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
                            
    if len(args.lr_ratio) == 0 or len(args.lr_schedule) == 0:
        lr_decay = LRDecay(mode='stair', lr=args.lr,)
    else:
        args.lr_schedule = [i * step_per_epoch for i in args.lr_schedule]
        lr_decay = LRDecay(mode=args.lr_decay_strat, lr=args.lr,
                           schedule=zip(args.lr_schedule, args.lr_ratio),
                           power_p=args.lr_decay_power,)
    
    ########################## load weight if specified
    if args.resume:
        resume_epoch = load_ckpt_epoch(args.resume,
                                [('model', net)],
                                [('optimizer', optimizer)],
                                strict=True)
        print('Model resume from', resume_epoch, 'epoch')
    elif args.init_ckpt:
        load_ckpt_epoch(args.load,[('model', net)], strict=True)
        print('Model loaded from', args.init_ckpt)
    elif args.init_pretrain:
        saved_state_dict = torch.load(args.init_pretrain)
        trans_state_dict = {}
        for i in saved_state_dict.keys():
            if 'conv1.' in i[:7]:
                conv1_weight = saved_state_dict[i]
                if conv1_weight.shape[1] != 10:
                    conv1_weight_mean = torch.mean(conv1_weight, dim=1, keepdim=True)
                    weight = (conv1_weight_mean / 10.0).repeat(1, 10, 1, 1)
            else:
                weight = saved_state_dict[i]
            trans_state_dict['module.encoder.'+i] = weight
        missing_keys, unexpected_keys = net.load_state_dict(trans_state_dict, strict=False)
        print ('Missing keys (the variable is in the graph but not in the dict):')
        print (missing_keys)
        print (' ')
        print ('Unexpected keys (the variable is in the dict but not in the graph): {}/{}'\
                .format(len(unexpected_keys), len(saved_state_dict)))
        print (unexpected_keys)
        print (' ')
        print ('Model loaded from', args.init_pretrain)
    else:
        print ('No weight provided, use random initialization.')
    
    ########################## setting up dataflow
    if args.size is None:
        train_dataset = DIMHeavyComposeSet(args.data, silence=True)
        eval_dataset = DIMHeavyComposeSet(args.data, isTrain=False, silence=True)
        size = (256, 256)
    else:
        size = args.size.split('x')
        size = (int(size[1]), int(size[0]))
        train_dataset = DIMHeavyComposeSet(args.data, size, silence=True)
        eval_dataset = DIMHeavyComposeSet(args.data, size, isTrain=False, silence=True)
    
    eval_loader = DataLoader(eval_dataset,
                             batch_size=torch.cuda.device_count(),
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=args.n_threads)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.n_threads,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
                              
    step_per_epoch = len(train_loader)
    print ('Will train for {} epochs, {} steps per epoch.'.format(args.max_epoch, step_per_epoch))

    ########################## main train loop
    loss = {}
    start_epoch = 0 if not args.resume else resume_epoch
    
    for epoch in range(start_epoch, args.max_epoch):
        print ('Start epoch {}...'.format(epoch+1))
        np.random.seed(BASE_SEED + epoch * args.n_threads)
        vis_loss = collections.OrderedDict()
        loss = {}
        
        ### training
        print ('Start training process...')
        net.train() # mode switch
        mean_loss = {'L_alpha':0., 'L_comp':0., 'L_grad':0., 'L_total':0.}
        with tqdm(train_loader) as t:
            for _step, dp in enumerate(t):
                step = epoch * step_per_epoch + _step
                t.set_description('{}/{}'.format(step, args.max_epoch * step_per_epoch))
                
                out = forward(net, BG_sample_net, FG_sample_net, dp, loss)
                loss = out[-1]
                loss_total = loss['L_total']
                # lr decay
                lr = adjust_learning_rate(optimizer, step, lr_decay)

                assert not np.isnan(loss_total.detach().cpu().numpy().squeeze()), "Loss is nan"

                # do optimize for current step
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                # write summaries, print log, save model
                for key in sorted(loss.keys()):
                    loss[key] = loss[key].detach().cpu().numpy().squeeze()
                    vis_loss[key] = '{:.4f}'.format(loss[key])
                    mean_loss[key] += loss[key]
                t.set_postfix(vis_loss)
                write_loss_dict(loss, writer, step, prefix='train/')
                writer.add_scalar('learning_rate', lr, step)
                write_image(out, writer, step, prefix='train/')
        
        for key in mean_loss.keys():
            mean_loss[key] /= float(step_per_epoch)
        print('='*(len(exp_name)+20))
        print(exp_name, "Epoch {} Training".format(epoch+1))
        print('='*(len(exp_name)+20))
        print_loss_dict(mean_loss)
        write_loss_dict(mean_loss, writer, epoch+1, prefix='train_mean/loss_')
        save_fn = os.path.join(model_save_dir, 'LSM_%d.pth' % (epoch+1))
        save_ckpt_epoch(save_fn, [('model', net)], [('optimizer', optimizer)], epoch+1)
        print('Model has been saved at '+save_fn)
        
        ### evaluation
        print ('Start evaluation process...')
        net.eval() # mode switch
        eval_loss = {'L_alpha':0., 'L_comp':0., 'L_grad':0., 'L_total':0.}
        with torch.no_grad():
            with tqdm(eval_loader) as t:
                for _step, dp in enumerate(t):
                    step = epoch * step_per_epoch + _step
                    
                    out = forward(net, BG_sample_net, FG_sample_net, dp, loss)
                    loss = out[-1]
                    for key in sorted(loss.keys()):
                        loss[key] = loss[key].detach().cpu().numpy().squeeze()
                        vis_loss[key] = '{:.4f}'.format(loss[key])
                        eval_loss[key] += loss[key]
                    
                    t.set_postfix(vis_loss)
                    write_image(out, writer, step, prefix='eval/')
        for key in eval_loss.keys():
            eval_loss[key] /= float(step_per_epoch)
        print('='*(len(exp_name)+20))
        print(exp_name, "Epoch {} Evaluation".format(epoch+1))
        print('='*(len(exp_name)+20))
        print_loss_dict(eval_loss)
        write_loss_dict(eval_loss, writer, epoch+1, prefix='eval/loss_')
        print ('Epoch {} completed.\n'.format(epoch+1))
        
    writer.close()

if __name__ == '__main__':
    args = parser()
    main(args)
    print ('Training Completed.')