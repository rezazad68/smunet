#!/usr/bin/env python3
# encoding: utf-8
# Modified from https://github.com/Wangyixinxin/ACN
import torch
from torch.nn import functional as F
import numpy as np
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image 
import cv2
import torch.nn as nn

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency = 10, consistency_rampup = 20.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)
  
def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def dice_loss(input, target):
    """soft dice loss"""
    eps = 1e-7
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

def gram_matrix(input):
    a, b, c, d, e = input.size()
    features = input.view(a * b, c * d * e)
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d * e)

def get_style_loss(sf, sm):
    g_f = gram_matrix(sf)
    g_m = gram_matrix(sm)
    channels = sf.size(1)
    size     = sf.size(2)*sf.size(3) 
    sloss = torch.sum(torch.square(g_f-g_m)) / (4.0 * (channels ** 2) * (size ** 2))
    return sloss*0.0001

def unet_Co_loss(config, batch_pred_full, content_full, batch_y, batch_pred_missing, content_missing, sf, sm, epoch):
    loss_dict = {}
    loss_dict['ed_dc_loss']  = dice_loss(batch_pred_full[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['net_dc_loss'] = dice_loss(batch_pred_full[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_dc_loss']  = dice_loss(batch_pred_full[:, 2], batch_y[:, 2])  # enhance tumor
    
    loss_dict['ed_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['net_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 2], batch_y[:, 2])  # enhance tumor

    ## Dice loss predictions
    loss_dict['loss_dc'] = loss_dict['ed_dc_loss'] + loss_dict['net_dc_loss'] + loss_dict['et_dc_loss']
    loss_dict['loss_miss_dc'] = loss_dict['ed_miss_dc_loss'] + loss_dict['net_miss_dc_loss'] + loss_dict['et_miss_dc_loss']
    
    ## Consistency loss
    loss_dict['ed_mse_loss']  = F.mse_loss(batch_pred_full[:, 0], batch_pred_missing[:, 0], reduction='mean') 
    loss_dict['net_mse_loss'] = F.mse_loss(batch_pred_full[:, 1], batch_pred_missing[:, 1], reduction='mean') 
    loss_dict['et_mse_loss']  = F.mse_loss(batch_pred_full[:, 2], batch_pred_missing[:, 2], reduction='mean') 
    loss_dict['consistency_loss'] = loss_dict['ed_mse_loss'] + loss_dict['net_mse_loss'] + loss_dict['et_mse_loss']
    
    ## Content loss
    loss_dict['content_loss'] = F.mse_loss(content_full, content_missing, reduction='mean')
    
    ## Style loss
    sloss = get_style_loss(sf, sm)
    
    
    ## Weights for each loss the lamba values
    weight_content = float(config['weight_content'])
    weight_missing = float(config['weight_mispath'])
    weight_full    = 1 - float(config['weight_mispath'])
    
    weight_consistency = get_current_consistency_weight(epoch)
    loss_dict['loss_Co'] = weight_full * loss_dict['loss_dc'] + weight_missing * loss_dict['loss_miss_dc'] + \
                            weight_consistency * loss_dict['consistency_loss'] + weight_content * loss_dict['content_loss']+sloss
    
    return loss_dict

def get_losses(config):
    losses = {}
    losses['co_loss'] = unet_Co_loss
    return losses


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.Tensor(prediction)
        target = torch.Tensor(target)
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()

        return ((2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)).numpy()