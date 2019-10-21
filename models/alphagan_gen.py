import torch.nn as nn
import torch
import torch.nn.functional as F
from .resnet_models import *

ENCODER_DICT = {
'resnet50':ResNet50,
'resnet50-aspp':ResNet50_ASPP,
}

class DecoderBlock(nn.Module):
    def __init__(self, input, outputs, kernels=[3, 3, 3], stride=1, output_bnrelu=True):
        super(DecoderBlock, self).__init__()
        assert len(outputs) == len(kernels)
        assert len(outputs) == 3
        self.output_bnrelu = output_bnrelu
        self.conv1 = nn.ConvTranspose2d(input, outputs[0], kernels[0], padding=1)
        self.bn1 = nn.BatchNorm2d(outputs[0])
        
        self.conv2 = nn.ConvTranspose2d(outputs[0], outputs[1], kernels[1], padding=1)
        self.bn2 = nn.BatchNorm2d(outputs[1])
        
        self.conv3 = nn.ConvTranspose2d(outputs[1], outputs[2], kernels[2], padding=1) \
            if stride == 1 else \
                nn.ConvTranspose2d(outputs[1], outputs[2], kernels[2], padding=1, \
                                   stride=stride, output_padding=1)
        if output_bnrelu:
            self.bn3 = nn.BatchNorm2d(outputs[2])
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        if self.output_bnrelu:
            x = F.relu(self.bn3(x))
        return x
 
class AlphaGANDecoder(nn.Module):
    def __init__(self, input, rgb_input):
        super(AlphaGANDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2., mode='bilinear', align_corners=True)
        self.skip2_1x1 = nn.Conv2d(input, 48, 1)
        self.skip2_bn = nn.BatchNorm2d(48)
        
        self.skip1_1x1 = nn.Conv2d(64, 32, 1)
        self.skip1_bn = nn.BatchNorm2d(32)
        
        self.skip0_3x3 = nn.Conv2d(rgb_input, 3, 3, padding=1)
        self.skip0_bn = nn.BatchNorm2d(3)
        
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.deconv1 = DecoderBlock(256 + 48, [256, 128, 64])
        self.deconv2 = DecoderBlock(64 + 32, [64, 64, 32], stride=2)
        self.deconv3 = DecoderBlock(32 + 3, [32, 32, 1], output_bnrelu=False)
        
    def forward(self, x, skip0, skip1, skip2, ind):
        x = self.upsample(x) # 2x upsample bottleneck
        skip2 = F.relu(self.skip2_bn(self.skip2_1x1(skip2)))
        x = torch.cat([x, skip2], axis=1)
        x = self.deconv1(x)
        x = self.unpool(x, ind)
        
        skip1 = F.relu(self.skip1_bn(self.skip1_1x1(skip1)))
        x = torch.cat([x, skip1], axis=1)
        x = self.deconv2(x)
        
        skip0 = F.relu(self.skip0_bn(self.skip0_3x3(skip0)))
        x = torch.cat([x, skip0], axis=1)
        x = self.deconv3(x)
        
        x = F.sigmoid(x)
        return x
        
def _AlphaGANDecoder(input, rgb_input):
    model = AlphaGANDecoder(input, rgb_input)
    return model

        
DECODER_DICT = {
'alphagan-decoder':_AlphaGANDecoder,
}

class AlphaGANGenerator(nn.Module):
    def __init__(self, input, encoder, decoder,
                 freeze_bn=False, freeze_dropout=False):
        super(AlphaGANGenerator, self).__init__()
        # input: RGB FG BG trimap/initial alpha matte? 3*3+1=10
        # make sure encoder always output 256 channel
        self.encoder = ENCODER_DICT[encoder](input)
        self.decoder = DECODER_DICT[decoder](256, input)
        self.freeze_bn = freeze_bn
        self.freeze_dropout = freeze_dropout
        
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or \
               isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) or \
                 isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        # x should be a 10-channel tensor by default
        # x = torch.cat([rgb, fg, bg, tri], axis=1)
        ind, skip1, skip2, bottleneck = self.encoder(x)
        res = self.decoder(bottleneck, x, skip1, skip2, ind)
        return res
        
    def train(self, mode=True):
        super(AlphaGANGenerator, self).train(mode=mode)
        ms = self.modules()
        if self.freeze_dropout:
            print("Freezing Dropout.")
            for m in ms:
                if isinstance(m, nn.Dropout):
                    m.eval()
        if self.freeze_bn:
            print("Freezing BatchNorm.")
            for m in ms:
                if isinstance(m, nn.BatchNorm2d) or \
                   isinstance(m, nn.SyncBatchNorm):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        