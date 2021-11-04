from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        # nn.ReLU(inplace=False)
        # nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        

        layers_conv_down = [nn.Conv2d(in_channels = input_size, out_channels= 32, kernel_size = (1, 1)), 
                            nn.Conv2d(in_channels = 32, out_channels= 64, kernel_size = (1, 1)), 
                            nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size = (1, 1)), 
                            nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size = (1, 1))]
                            

        layers_bn_down = [nn.BatchNorm2d(num_features= 32), nn.BatchNorm2d(num_features= 64), nn.BatchNorm2d(num_features= 128), nn.BatchNorm2d(num_features= 256)]
        
        layers_pooling = [nn.MaxPool2d(kernel_size= 2, return_indices=True), nn.MaxPool2d(kernel_size= 2, return_indices=True), nn.MaxPool2d(kernel_size= 2, return_indices=True), nn.MaxPool2d(kernel_size= 2, return_indices=True)]


        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU

        layers_conv_down = [nn.Conv2d(in_channels = input_size, out_channels= 32, kernel_size = (1, 1)), 
                            nn.Conv2d(in_channels = 32, out_channels= 64, kernel_size = (1, 1)), 
                            nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size = (1, 1)), 
                            nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size = (1, 1))]
                            

        layers_bn_down = [nn.BatchNorm2d(num_features= 32), nn.BatchNorm2d(num_features= 64), nn.BatchNorm2d(num_features= 128), nn.BatchNorm2d(num_features= 256)]
        
        layers_pooling = [nn.MaxPool2d(kernel_size= (2,2), return_indices=True), nn.MaxPool2d(kernel_size= (2,2), return_indices=True), nn.MaxPool2d(kernel_size= (2,2), return_indices=True), nn.MaxPool2d(kernel_size= (2,2), return_indices=True)]

        layers_conv_up = [nn.Conv2d(in_channels = 256, out_channels= 128, kernel_size = (1, 1)),
                        nn.Conv2d(in_channels = 128, out_channels= 64, kernel_size = (1, 1)),
                        nn.Conv2d(in_channels = 64, out_channels= 32, kernel_size = (1, 1)),
                        nn.Conv2d(in_channels = 32, out_channels= 32, kernel_size = (1, 1))
        ]
        layers_bn_up = [nn.BatchNorm2d(num_features= 256), nn.BatchNorm2d(num_features= 128), nn.BatchNorm2d(num_features= 64), nn.BatchNorm2d(num_features= 32)]
        layers_unpooling = [nn.MaxUnpool2d(2), nn.MaxUnpool2d(2), nn.MaxUnpool2d(2), nn.MaxUnpool2d(2)]


        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)

        self.final_conv = nn.Conv2d(in_channels = 32, out_channels= 11, kernel_size = (1, 1)),

    def forward(self, x):
        indices = []
        relu_out = x
        for i in range(4):
            conv_output = self.layers_conv_down[i](relu_out)
            pooled_out, temp_indices = self.layers_pooling[i](conv_output)
            indices += [temp_indices]
            normalised_out = self.layers_bn_down(pooled_out)
            relu_out = self.relu(normalised_out)
        
        for i in range(3, -1, -1):
            pooled_out = self.layers_unpooling[i](relu_out)
            conv_output = self.layers_conv_up[i](pooled_out, indices[i])

            normalised_out = self.layers_bn_up(pooled_out)
            relu_out = self.relu(normalised_out)

        return self.final_conv(relu_out)
        
            

def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
