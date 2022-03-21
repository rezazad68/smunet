#!/usr/bin/env python3
# encoding: utf-8
from .unet import Unet_module

def build_model(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = Unet_module(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)

    model_missing = Unet_module(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)
    return model_full, model_missing
