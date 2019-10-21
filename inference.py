import sys, os
import argparse
import collections
import datetime
import multiprocessing as mp
import torch
import numpy as np
import cv2
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils.loss_func as L
from dataset.DeepMattingEval import DIMEvaluationDataset
from models.alphagan_gen import \
    AlphaGANGenerator, ENCODER_DICT, DECODER_DICT
from models import DeepFill
from utils.io import *
from utils.runner_func import *

BASE_SEED = 7777777    

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use. Default to use all available ones')
    parser.add_argument('--eval', required=True, help='Path to dataset')
    parser.add_argument('--out', required=True, help='Path to output folder')
    parser.add_argument('--n_threads', type=int, default=16, help='number of workers for dataflow.')
    parser.add_argument('-d', '--decoder', default='alphagan-decoder', choices=DECODER_DICT.keys())
    parser.add_argument('-e', '--encoder', default='resnet50-aspp', choices=ENCODER_DICT.keys())
    parser.add_argument('--bg_model_weight', default='BGDFV1.pth')
    parser.add_argument('--fg_model_weight', default='FGDFV1.pth')
    parser.add_argument('--mat_model_weight', default='LSM_80.pth')
    parser.add_argument('-s', '--size', default='800x800', \
        help='the size of the network input. Format: WxH (Default: 800x800)')
    args = parser.parse_args()
    return args
    
def forward(net, BG, FG, dp,):
    def _sampling_forward(dp):
        rgb, tri, small_tri, idx = dp
        # rgb, fg_gt, bg_gt is in [-1, 1]
        # gt is in [0, 1]
        # tri is 1-channel trimap in {0, 1, 2}, 0 BG 1 UN 2 FG
        # preprocess which does not need gradients
        rgb = rgb.cuda().float().clamp(-1., 1.) # [b, 3, h, w]
        tri = tri.cuda().float().clamp(0., 2.)  # [b, 1, h, w]
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
        return fg_, bg_, rgb_, fg_pred, bg_pred, rgb, tri, mask, idx
    
    fg_, bg_, rgb_, fg_pred, bg_pred, rgb, tri, mask, idx = _sampling_forward(dp)
    input_x = torch.cat([rgb, fg_pred, bg_pred, tri-1.], axis=1)
    pred = net(input_x)        
    alpha = torch.where(mask, pred, tri / 2.0)
    # composition
    comp = fg_ * alpha + bg_ * (1. - alpha)
    
    return [rgb_, tri, fg_, bg_, alpha, comp, mask, idx]

def main(args):
    ########################## set cuda environment
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    print ('Write to path:', args.out)
    
    ########################## random seed initialization
    torch.manual_seed(BASE_SEED)
    # ONLY USE THESE TWO LINES FOR DEBUGGING
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    ########################## init net, deal with sync_bn
    net = AlphaGANGenerator(input=10,
                            encoder=args.encoder, decoder=args.decoder)
    net = nn.DataParallel(net).cuda()
    net.eval()
    
    ########################## build FG / BG and load weight for them
    BG_sample_net = DeepFill.Generator(build_refine=True)
    FG_sample_net = DeepFill.Generator(build_refine=True, is_fg=True)
    BG_sample_net = nn.DataParallel(BG_sample_net).cuda()
    FG_sample_net = nn.DataParallel(FG_sample_net).cuda()
    BG_sample_net.eval()
    FG_sample_net.eval()
    load_ckpt_epoch(args.mat_model_weight,[('model', net)], strict=True,
                    conv1_chan=10)
    print ('Matting net loaded weight from '+args.mat_model_weight)
    load_ckpt_epoch(args.bg_model_weight, [('model', BG_sample_net)], strict=True)
    print ('BG sampling net loaded weight from '+args.bg_model_weight)
    load_ckpt_epoch(args.fg_model_weight, [('model', FG_sample_net)], strict=True)
    print ('FG sampling net loaded weight from '+args.fg_model_weight)
    
    ########################## setting up dataflow
    size = [int(i) for i in args.size.split('x')] if args.size is not None else [800, 800]
    assert size[0] % 32 == 0 and size[1] % 32 == 0
    eval_dataset = DIMEvaluationDataset(args.eval, input_shape=(size[1], size[0]))
    img_shape = eval_dataset.img_shape
    imgpath = eval_dataset.imgpath
    
    print ('Network input shape:', eval_dataset.input_shape)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=torch.cuda.device_count(),
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=args.n_threads)
                              
    ### evaluation
    def _write_image(tensor, idx, suffix, resize=True):
        x = tensor.cpu().numpy().transpose([1, 2, 0])
        if resize:
            x = cv2.resize(x, dsize=(img_shape[idx][1], img_shape[idx][0]), interpolation=cv2.INTER_AREA)
        fn = os.path.basename(imgpath[idx][:imgpath[idx].rfind('_')])
        fn = os.path.join(args.out, '{}_{}.png'.format(fn, suffix))
        cv2.imwrite(fn, np.uint8(x*255))
    print ('Start evaluation process...')
    net.eval() # mode switch
    with torch.no_grad():
        with tqdm(eval_loader) as t:
            for _step, dp in enumerate(t):
                out = forward(net, BG_sample_net, FG_sample_net, dp)
                rgb_, tri, fg_, bg_, alpha, comp, mask, idx = out
                idx = idx.cpu().numpy()
                for i, id in enumerate(idx):
                    _write_image(fg_[i], id, 'fg')
                    _write_image(bg_[i], id, 'bg')
                    _write_image(alpha[i], id, 'alpha')
                    _write_image(comp[i], id, 'rgb')
                    _write_image(tri[i] / 2.0, id, 'trimap')
    print ('Completed.')

if __name__ == '__main__':
    args = parser()
    main(args)