import numpy as np
import pandas as pd

from Datasets.utk_face import UTKFaceDataset

#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

def reshape(shape):
    return lambda x : x.reshape(shape)

inverse_layer_map = {
    nn.ReLU:                        lambda          **kwargs : nn.ReLU(),
    nn.LeakyReLU:                   lambda x,       **kwargs : nn.LeakyReLU(x.negative_slope),
    nn.Linear:                      lambda x,       **kwargs : nn.Linear(x.out_features, x.in_features),
    nn.LazyLinear:                  lambda x,shape, **kwargs : nn.Linear(x.out_features, np.prod(shape)),
    nn.Conv2d:                      lambda x,       **kwargs : nn.ConvTranspose2d(x.out_channels, x.in_channels, x.kernel_size, x.stride),
    nn.Flatten:                     lambda shape,   **kwargs : nn.Unflatten(1, shape[1:])
}
def invert(layers, input_shape=(1, 3, 200, 200)):
    example = torch.Tensor(np.zeros(input_shape))
    inverse_layers = []
    for layer in layers:
        inverse = inverse_layer_map[type(layer)](x=layer, shape=example.shape)
        inverse_layers.append(inverse)

        example = layer.to('cpu')(example)

    return reversed(inverse_layers), example.shape
    
class Autoencoder(nn.Module):
    def __init__(self, layers):
        super(Autoencoder, self).__init__()
        inverse_layers, self.z_shape = invert(layers)

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(*inverse_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x