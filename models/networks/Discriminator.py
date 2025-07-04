"""
Discriminator

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#logging module with handmade settings.
from CaloQVAE import logging
logger = logging.getLogger(__name__)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, dim=64, out_conv_channels=512):
        super(Discriminator, self).__init__()
        conv1_channels = int(out_conv_channels / 8)
        conv2_channels = int(out_conv_channels / 4)
        conv3_channels = int(out_conv_channels / 2)
        self.out_conv_channels = out_conv_channels
        self.out_dim = int(dim / 16)
        self.z = 45
        self.r = 9
        self.phi = 16

        self.conv1 = nn.Sequential(
            # nn.Conv3d(
            #     in_channels=in_channels, out_channels=conv1_channels, kernel_size=(4,4,2),
            #     stride=2, padding=1, bias=False
            # ),
            nn.Conv3d(
            in_channels=in_channels, out_channels=conv1_channels, kernel_size=(7,4,3),
            stride=(2,2,2), padding=0, bias=False
            ),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            # nn.Conv3d(
            #     in_channels=conv1_channels+1, out_channels=conv2_channels, kernel_size=(4,4,3),
            #     stride=2, padding=1, bias=False
            # ),
            nn.Conv3d(
            in_channels=conv1_channels+1, out_channels=conv2_channels, kernel_size=(7,4,3),
            stride=(2,2,1), padding=0, bias=False
            ),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            # nn.Conv3d(
            #     in_channels=conv2_channels, out_channels=out_conv_channels, kernel_size=(4,4,3),
            #     stride=2, padding=1, bias=False
            # ),
            nn.Conv3d(
            in_channels=conv2_channels, out_channels=out_conv_channels, kernel_size=(6,2,2),
            stride=(2,1,1), padding=0, bias=False
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv3d(
        #         in_channels=conv3_channels, out_channels=out_conv_channels, kernel_size=(6,4,3),
        #         stride=2, padding=1, bias=False
        #     ),
        #     nn.BatchNorm3d(out_conv_channels),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        self.out = nn.Sequential(
            nn.Linear(out_conv_channels , 1),
            # nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, x0):
        x = x.reshape(x.shape[0], 1, self.z, self.phi, self.r) 
        x = self.conv1(x)
        
        x0 = self.trans_energy(x0)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # Flatten and apply linear + sigmoid
        x = x.view(-1, self.out_conv_channels)
        x = self.out(x)
        return x
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map