"""
Decoder

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import LeakyReLU, ReLU

# from models.networks.networks import Network, NetworkV2, NetworkV3
from models.networks.basicCoders import BasicDecoderV3

from utils.dists.gumbelmod import GumbelMod

#logging module with handmade settings.
# from CaloQVAE import logging
# logger = logging.getLogger(__name__)

class sequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class DecoderCNNPB(BasicDecoderV3):
# class DecoderCNNPB(nn.Module):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPB, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes

        self.n_latent_nodes = self._config.model.n_latent_nodes
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]

        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),

                   nn.ConvTranspose2d(self._node_sequence[0][0]-1, 1028, (3,5), 2, 0),
                   nn.BatchNorm2d(1028),
                   nn.PReLU(1028, 0.02),
                   

                   nn.ConvTranspose2d(1028, 512, (3,5), 1, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   nn.ConvTranspose2d(513, 128, (3,5), 1, 0),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),

                   nn.ConvTranspose2d(128, 45, (3,4), 1, 0),
                   # nn.BatchNorm2d(45),
                   nn.PReLU(45, 0.02),
                                   )
        
        self._layers3 = nn.Sequential(
                   nn.ConvTranspose2d(513, 128, (3,5), 1, 0),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),

                   nn.ConvTranspose2d(128, 45, (3,4), 1, 0),
                   # nn.BatchNorm2d(45),
                   nn.PReLU(45, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000)), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],45*9*16), x2.reshape(x1.shape[0],45*9*16)
    
    
class DecoderCNNPBv2(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPBv2, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.z = 45
        self.r = 9
        self.phi = 16

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.n_latent_nodes, 1,1)),

                   PeriodicConvTranspose2d(self.n_latent_nodes, 1024, (3,5), 2, 0),
                   nn.BatchNorm2d(1024),
                   nn.PReLU(1024, 0.02),
                   

                   PeriodicConvTranspose2d(1024, 512, (3,5), 1, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose2d(513, 128, (3,5), 1, 0),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),

                   PeriodicConvTranspose2d(128, 45, (3,4), 1, 0),
                   nn.BatchNorm2d(45),
                   nn.PReLU(45, 1.0),
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose2d(513, 128, (3,5), 1, 0),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),

                   PeriodicConvTranspose2d(128, 45, (3,4), 1, 0),
                   nn.BatchNorm2d(45),
                   nn.PReLU(45, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000)), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    
class DecoderCNNPBv3(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPBv3, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.z = 45
        self.r = 9
        self.phi = 16

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.n_latent_nodes, 1,1)),

                   PeriodicConvTranspose2d(self.n_latent_nodes, 1024, (2,2), 2, 0),
                   nn.BatchNorm2d(1024),
                   nn.PReLU(1024, 0.02),
                   

                   PeriodicConvTranspose2d(1024, 512, (2,5), 1, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
    
                   PeriodicConvTranspose2d(512, 256, (3,5), 1, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose2d(257, 128, (2,2), 1, 0),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),

                   PeriodicConvTranspose2d(128, 64, (2,2), 1, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 1.0),
    
                   PeriodicConvTranspose2d(64, 45, (3,5), 1, 0),
                   nn.BatchNorm2d(45),
                   nn.PReLU(45, 1.0),
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose2d(257, 128, (2,2), 1, 0),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),

                   PeriodicConvTranspose2d(128, 64, (2,2), 1, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 1.0),
    
                   PeriodicConvTranspose2d(64, 45, (3,5), 1, 0),
                   nn.BatchNorm2d(45),
                   nn.PReLU(45, 1.0),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000)), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
class DecoderCNNPBv4(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPBv4, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.z = 45
        self.r = 9
        self.phi = 16

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.n_latent_nodes, 1,1)),

                   PeriodicConvTranspose2d(self.n_latent_nodes, 1024, (3,5), 2, 0),
                   nn.BatchNorm2d(1024),
                   nn.PReLU(1024, 0.02),
                   

                   PeriodicConvTranspose2d(1024, 512, (3,4), 1, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose2d(513, 128, (3,3), 1, 1),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),

                   PeriodicConvTranspose2d(128, 45, (3,3), 1, 1),
                   # nn.BatchNorm2d(45),
                   nn.PReLU(45, 1.0),
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose2d(513, 128, (3,3), 1, 1),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),

                   PeriodicConvTranspose2d(128, 45, (3,3), 1, 1),
                   # nn.BatchNorm2d(45),
                   nn.PReLU(45, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0):
        return (torch.log(x0) - log_e_min)/(log_e_max - log_e_min)
    
    
class PeriodicConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConvTranspose2d, self).__init__()
        self.padding = padding
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Pad input tensor with periodic boundary conditions
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')
        # Apply convolution
        x = self.conv(x)
        return x
    
####################################
class PeriodicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConv3d, self).__init__()
        self.padding = padding
        # try 3x3x3 cubic convolution
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        # Pad input tensor with periodic boundary and circle-center conditions
        if self.padding == 1:
            mid = x.shape[-1] // 2
            shift = torch.cat((x[..., [-1], mid:], x[..., [-1], :mid]), -1)
            x = torch.cat((x, shift), dim=-2)
        x = F.pad(x, (self.padding, self.padding, 0, 0, 0, 0), mode='circular')
        # Apply convolution
        x = self.conv(x)
        return x

class PeriodicConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConvTranspose3d, self).__init__()
        self.padding = padding
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        # Pad input tensor with periodic boundary conditions
        if self.padding == 1:
            mid = x.shape[-2] // 2
            shift = torch.cat((x[..., mid:, [0]], x[..., :mid, [0]]), -2)
            x = torch.cat((shift,x), dim=-1)
            x = F.pad(x, (0, 0, self.padding, self.padding, 0, 0), mode='circular')
        return x

class DecoderCNNPB3Dv1(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPB3Dv1, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.z = 45
        self.r = 9
        self.phi = 16

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.n_latent_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.n_latent_nodes, 512, (3,3,3), (2,1,2), 0),
                   nn.BatchNorm3d(512),
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,3,4), (2,1,1), 1),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (5,3,3), (2,1,1), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 1, (5,3,3), (2,1,1), 1),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 1.0),
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (5,3,3), (2,1,1), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 1, (5,3,3), (2,1,1), 1),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 0.02),
                                   )
        
    def forward(self, x, x0):
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 15 * 1.2812657528661318):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map

class DecoderCNNPB3Dv2(BasicDecoderV3): #use this one
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPB3Dv2, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.z = 45
        self.r = 9
        self.phi = 16

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4

        # dropout for regularization
        # self.dropout = nn.Dropout3d(self._config.model.dropout_prob)
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.n_latent_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.n_latent_nodes, 512, (3,2,3), (2,1,1), 0),
                   nn.BatchNorm3d(512),
                   # self.dropout,
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,3,3), (2,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,2,3), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,1,2), 0),
                   nn.BatchNorm3d(32),
                   # self.dropout,
                   nn.PReLU(32, 1.0),

                   PeriodicConvTranspose3d(32, 1, (5,3,2), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,2,3), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,1,2), 0),
                   nn.BatchNorm3d(32),
                   # self.dropout,
                   nn.PReLU(32, 0.02),

                   PeriodicConvTranspose3d(32, 1, (5,3,2), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 15 * 1.2812657528661318):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    
class DecoderCNNPB3Dv3(BasicDecoderV3): #use this one
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPB3Dv3, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.z = 45
        self.r = 9
        self.phi = 16

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4

        # dropout for regularization
        # self.dropout = nn.Dropout3d(self._config.model.dropout_prob)
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.n_latent_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.n_latent_nodes, 512, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(512),
                   # self.dropout,
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,3,3), (2,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,2,1), 0),
                   nn.BatchNorm3d(32),
                   # self.dropout,
                   nn.PReLU(32, 1.0),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,2,1), 0),
                   nn.BatchNorm3d(32),
                   # self.dropout,
                   nn.PReLU(32, 0.02),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    
class DecoderCNNPB3Dv3Reg(BasicDecoderV3): #use this one
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPB3Dv3Reg, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.z = self._config.data.z #45
        self.r = self._config.data.r #9
        self.phi = self._config.data.phi #16

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4

        # dropout for regularization
        # self.dropout = nn.Dropout3d(self._config.model.dropout_prob)
        
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.n_latent_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.n_latent_nodes, 512, (3,3,3), (2,2,2), 0),
                   nn.BatchNorm3d(512),
                   # self.dropout,
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (3,3,3), (1,2,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (2,3,2), (1,1,2), 1),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 1, (2,2,3), (1,1,2), 1),
                   nn.BatchNorm3d(1),
                   # self.dropout,
                   nn.PReLU(1, 1.0),

                   # PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   # nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (2,3,2), (1,1,2), 1),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 1, (2,2,3), (1,1,2), 1),
                   nn.BatchNorm3d(1),
                   # self.dropout,
                   nn.PReLU(1, 0.02),

                   # PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   # nn.PReLU(1, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=16.0, log_e_min=5.0, s_map = 1.0):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    
class DecoderCNNPB3Dv5(BasicDecoderV3): #use this one
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPB3Dv5, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.z = 45
        self.r = 9
        self.phi = 16

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4

        # dropout for regularization
        self.dropout = nn.Dropout3d(self._config.model.dropout_prob)
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.n_latent_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.n_latent_nodes, 512, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(512),
                   # self.dropout,
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,3,3), (2,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(128, 64, (3,3,2), (2,1,1), 1),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,1,1), 1),
                   nn.BatchNorm3d(32),
                   # self.dropout,
                   nn.PReLU(32, 0.02),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 1),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(128, 64, (3,3,2), (2,1,1), 1),
                   nn.BatchNorm3d(64),
                   self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,1,1), 1),
                   nn.BatchNorm3d(32),
                   self.dropout,
                   nn.PReLU(32, 0.02),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 1),
                   # nn.BatchNorm3d(45),
                   self.dropout,
                   nn.PReLU(1, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = x + x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,x.shape[2],x.shape[3],x.shape[4])
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    

class DecoderCNNPBv4_HEMOD(BasicDecoderV3):
    def __init__(self, num_input_nodes, num_output_nodes, output_activation_fct=nn.Identity(), **kwargs):
        super(DecoderCNNPBv4_HEMOD, self).__init__(**kwargs)
        self._output_activation_fct = output_activation_fct
        self.num_input_nodes = num_input_nodes
        self.z = 45
        self.r = 9
        self.phi = 16
        self.hierarchal_outputs = num_output_nodes
        self.output_layers = int(self.hierarchal_outputs / 144)
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.num_input_nodes, 1,1)),

                   PeriodicConvTranspose2d(self.num_input_nodes, 1024, (3,5), 2, 0),
                   nn.BatchNorm2d(1024),
                   nn.PReLU(1024, 0.02),
                   

                   PeriodicConvTranspose2d(1024, 512, (3,4), 1, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose2d(513, 128, (3,3), 1, 1),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),

                   PeriodicConvTranspose2d(128, self.output_layers, (3,3), 1, 1),
                   # nn.BatchNorm2d(45),
                   nn.PReLU(self.output_layers, 1.0),
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose2d(513, 128, (3,3), 1, 1),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),

                   PeriodicConvTranspose2d(128, self.output_layers, (3,3), 1, 1),
                   # nn.BatchNorm2d(45),
                   nn.PReLU(self.output_layers, 0.02),
                                   )
        
    def forward(self, x, x0):
        # print("t1: ", x.shape)
        x = self._layers(x)
        # print("t2: ", x.shape)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        # need channels * height * width = self.hierarchal_outputs = 1620
        return x1.reshape(x1.shape[0], self.hierarchal_outputs), x2.reshape(x1.shape[0], self.hierarchal_outputs)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 15 * 1.2812657528661318):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map

class DecoderCNNPB3Dv1_HEMOD(BasicDecoderV3):
    def __init__(self, num_input_nodes, num_output_nodes, output_activation_fct=nn.Identity(), **kwargs):
        super(DecoderCNNPB3Dv1_HEMOD, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_input_nodes = num_input_nodes
        self.z = 45
        self.r = 9
        self.phi = 16
        self.hierarchal_outputs = num_output_nodes
        self.output_layers = int(self.hierarchal_outputs / 144)

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        # self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.num_input_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.num_input_nodes, 512, (3,2,3), (2,1,1), 0),
                   nn.BatchNorm3d(512),
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,3,3), (2,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,2,3), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   self.dropout,
                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,1,2), 0),
                   nn.BatchNorm3d(32),
                   nn.PReLU(32, 0.02),

                   PeriodicConvTranspose3d(32, 1, (5,3,2), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,2,3), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,1,2), 0),
                   nn.BatchNorm3d(32),
                   nn.PReLU(32, 0.02),

                   PeriodicConvTranspose3d(32, 1, (5,3,2), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0], self.hierarchal_outputs), x2.reshape(x1.shape[0], self.hierarchal_outputs)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 15 * 1.2812657528661318):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map


class DecoderCNNPB3Dv2_HEMOD(BasicDecoderV3):
    def __init__(self, num_input_nodes, num_output_nodes, output_activation_fct=nn.Identity(), **kwargs):
        super(DecoderCNNPB3Dv2_HEMOD, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_input_nodes = num_input_nodes
        self.z = 45
        self.r = 9
        self.phi = 16
        self.hierarchal_outputs = num_output_nodes
        self.output_layers = int(self.hierarchal_outputs / 144)

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        # self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4
        
        # dropout for regularization
        # self.dropout = nn.Dropout3d(self._config.model.dropout_prob) # config object is NoneType error?
        self.dropout = nn.Dropout3d(0.2)

        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.num_input_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.num_input_nodes, 512, (3,2,3), (2,1,1), 0),
                   nn.BatchNorm3d(512),
                   self.dropout,
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,3,3), (2,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,2,3), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,1,2), 0),
                   nn.BatchNorm3d(32),
                   self.dropout,
                   nn.PReLU(32, 0.02),

                   PeriodicConvTranspose3d(32, 1, (5,3,2), (1,1,1), 0),
                   PeriodicConv3d(1, 1, (self.z - self.output_layers + 1, 1, 1), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,2,3), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,1,2), 0),
                   nn.BatchNorm3d(32),
                   self.dropout,
                   nn.PReLU(32, 0.02),

                   PeriodicConvTranspose3d(32, 1, (5,3,2), (1,1,1), 0),
                   PeriodicConv3d(1, 1, (self.z - self.output_layers + 1, 1, 1), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0], self.hierarchal_outputs), x2.reshape(x1.shape[0], self.hierarchal_outputs)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 15 * 1.2812657528661318):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map


class DecoderCNNPB_HEv1(BasicDecoderV3):
    def __init__(self, encArch = 'Large', num_output_nodes=None, **kwargs):
        self.encArch = encArch
        super(DecoderCNNPB_HEv1, self).__init__(**kwargs)
        self._hit_smoothing_dist_mod = GumbelMod()
        self._inference_energy_activation_fct = ReLU()
        # self.device = x.device # add something like this to reduce computation time
        self._create_hierarchy_network()

    def _training_activation_fct(self, slope):
        return LeakyReLU(slope)

    def _create_hierarchy_network(self, level: int = 0):
        self.latent_nodes = self._config.model.n_latent_nodes_per_p * 4
        # change these variables for different HE decoder structures
        self.n_layers_per_subdec = 5
        self.layer_step = self.n_layers_per_subdec*144
         # varies depending on if last layer is > or < layer step
        self.hierarchical_lvls = 9

        inp_layers = [self.latent_nodes + i * self.layer_step for i in range(self.hierarchical_lvls)] 
        out_layers = self.hierarchical_lvls * [self.layer_step]

        out_layers[self.hierarchical_lvls - 1] += (6480 - self.hierarchical_lvls * self.layer_step)
        # print(self.raw_layers)

        # Unbalanced Hierachical Decoder
        # inp_layers[0:5] = [self.latent_nodes]
        # out_layers[0:5] = [sum(out_layers[0:5])]
        self.raw_layers = [layers - self.latent_nodes for layers in inp_layers] + [6480]
        
        # Check Layers
        print("Layer Inputs: ", inp_layers)
        print("Layer Outputs: ", out_layers)
        print("Raw Layer Indices: ", self.raw_layers)

        self.moduleLayers = nn.ModuleList([])
        for i in range(len(inp_layers)):
            # self.moduleLayers.append(DecoderCNNPBv4_HEMOD(inp_layers[i], out_layers[i]))
            # self.moduleLayers.append(DecoderCNNPB3Dv1_HEMOD(inp_layers[i], out_layers[i]))
            self.moduleLayers.append(DecoderCNNPB3Dv2_HEMOD(inp_layers[i], out_layers[i]))

        # not used
        sequential = sequentialMultiInput(*self.moduleLayers)
        return sequential
    
    def forward(self, x, x0, act_fct_slope, x_raw):
        self.sub_values = []
        self.x1, self.x2 = torch.tensor([]).to(x.device), torch.tensor([]).to(x.device) # store hits and activation tensors
        # Instead of in range(self.hierarchical_lvls) just use len(self.moduleLayers) to deal with unbalanced hierarchical decoders
        for lvl in range(len(self.moduleLayers)):
            cur_net = self.moduleLayers[lvl]
            output_hits, output_activations = cur_net(x, x0)
            beta = torch.tensor(self._config.model.output_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
            # if self._config.engine.modelhits:
            if self.training:
                # out.output_activations = self._energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
                activation_fct_annealed = self._training_activation_fct(act_fct_slope)
                # out.output_activations = activation_fct_annealed(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
                # print(self.raw_layers[lvl+1])
                outputs = activation_fct_annealed(output_activations) * torch.where(x_raw[:, self.raw_layers[lvl]:self.raw_layers[lvl+1]] > 0, 1., 0.)
            else:
                outputs = self._inference_energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training=False)
            z = outputs
            self.sub_values.append([output_hits, output_activations])
            if lvl == len(self.moduleLayers) - 1:
                for vals in self.sub_values:
                    self.x1 = torch.cat((self.x1, vals[0]), dim=1)
                    self.x2 = torch.cat((self.x2, vals[1]), dim=1)
            else:
                x = torch.cat((x, z), dim=1)
        return self.x1, self.x2
    
    
class PeriodicConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConvTranspose2d, self).__init__()
        self.padding = padding
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Pad input tensor with periodic boundary conditions
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')
        # Apply convolution
        x = self.conv(x)
        return x
    
    
##############LinAtt
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=32, cylindrical = False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        if(cylindrical):
            # self.to_qkv = CylindricalConv(dim, hidden_dim * 3, kernel_size = 1, bias=False)
            # self.to_out = nn.Sequential(CylindricalConv(hidden_dim, dim, kernel_size = 1), nn.GroupNorm(1,dim))
            self.to_qkv = PeriodicConv3d(dim, hidden_dim * 3, kernel_size = 1, bias=False)
            self.to_out = nn.Sequential(PeriodicConv3d(hidden_dim, dim, kernel_size = 1), nn.GroupNorm(1,dim))
        else: 
            self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, kernel_size = 1, bias=False)
            self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, kernel_size = 1), nn.GroupNorm(1,dim))

    def forward(self, x):
        b, c, l, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=l, y=h, z = w)
        return self.to_out(out)


class DecoderCNNPB3Dv4(BasicDecoderV3): #use this one
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPB3Dv4, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.z = 45
        self.r = 9
        self.phi = 16

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4

        # dropout for regularization
        # self.dropout = nn.Dropout3d(self._config.model.dropout_prob)
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.n_latent_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.n_latent_nodes, 512, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(512),
                   # self.dropout,
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,3,3), (2,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,2,1), 0),
                   nn.BatchNorm3d(32),
                   # self.dropout,
                   nn.PReLU(32, 1.0),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 0),
                   nn.GroupNorm(1,64),
                   # self.dropout,
                   nn.SiLU(64),
                   # Residual(PreNorm(64, LinearAttention(64, cylindrical = False))),
                   LinearAttention(64, cylindrical = False),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,2,1), 0),
                   nn.GroupNorm(1,32),
                   # self.dropout,
                   nn.SiLU(32),
                   LinearAttention(32, cylindrical = False),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.SiLU(1),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map

class LNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(self.fn(x))
        return x

class Head(nn.Module):
    '''
    Self-attention block
    '''
    def __init__(self, dim, head_size=16):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(dim,head_size, bias=False)
        self.query = nn.Linear(dim,head_size, bias=False)
        self.value = nn.Linear(dim,head_size, bias=False)
        # self.ln1 = nn.LayerNorm(dim)
        # self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, l, h, w = x.shape
        x = rearrange(x, "b c l h w -> b (l h w) c")
        # x = self.ln1(x)
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5
        wei = F.softmax(wei,dim=-1)
        v = self.value(x)
        # out = self.ln2(wei @ v)
        out = wei @ v
        
        return rearrange(out, "b (l h w) c -> b c l h w",l=l,h=h,w=w)

    
class Multihead(nn.Module):
    '''
        Multi-head attention
    '''
    def __init__(self, dim, num=1, head_size=16):
        super().__init__()
        self.heads = nn.ModuleList([Head(dim,head_size) for _ in range(num)])
        
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=1)
    
class Headv2(nn.Module):
    '''
    Self-attention block
    '''
    def __init__(self, dim, head_size=16):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(dim,head_size, bias=False)
        self.query = nn.Linear(dim,head_size, bias=False)
        self.value = nn.Linear(dim,head_size, bias=False)
        

    def forward(self, x):
        # b, c, l, h, w = x.shape
        # x = rearrange(x, "b c l h w -> b (l h w) c")
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5
        wei = F.softmax(wei,dim=-1)
        v = self.value(x)
        out = wei @ v
        
        return out
    
class Multiheadv2(nn.Module):
    '''
        Multi-head attention
    '''
    def __init__(self, dim, num=1):
        super().__init__()
        head_size = dim // num
        self.heads = nn.ModuleList([Headv2(dim,head_size) for _ in range(num)])
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        b, c, l, h, w = x.shape
        x = rearrange(x, "b c l h w -> b (l h w) c")
        x = torch.cat([h(self.ln1(x)) for h in self.heads], dim=2)
        x = self.ln2(x)
        return rearrange(x, "b (l h w) c -> b c l h w",l=l,h=h,w=w)
    
    
class DecoderCNNPB3DSelfAtt(BasicDecoderV3): #use this one
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPB3DSelfAtt, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.z = 45
        self.r = 9
        self.phi = 16

        self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.n_latent_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.n_latent_nodes, 512, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(512),
                   # self.dropout,
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,3,3), (2,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,2,1), 0),
                   nn.BatchNorm3d(32),
                   # self.dropout,
                   nn.PReLU(32, 1.0),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 1.0)
                                   )
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,2,2), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 1.0),

                   PeriodicConvTranspose3d(64, 32, (3,3,3), (1,1,1), 0),
                   # nn.BatchNorm3d(32),
                   # Residual(Multihead(32,2,16)),
                   # Multihead(32,2,16),
                   Multiheadv2(32,2),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (2,2,1), 0),
                   # nn.BatchNorm3d(45),
                   # nn.PReLU(1,0.1),
                   nn.SiLU(1),
                   NoiseAdd(self._config.model.std_noise),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map

    
############################Cholesky
class NoiseAddCholesky(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        if self.training:
            bss = x.shape
            z = x.view(bss[0]*bss[1],-1).detach()
            cov = torch.cov(z.T)
            L = torch.linalg.cholesky((cov+cov.T)/2)
            # return x + torch.randn_like(z).to(x.device) @ L
            return x + (torch.randn_like(z).to(x.device) @ L).reshape(bss)
        else:
            return x
        
        
class NoiseAdd(nn.Module):
    def __init__(self, std=1.0):
        super().__init__()
        self.std = std

    def forward(self,x):
        # if self.training:
        return x + torch.randn_like(x).to(x.device) * self.std
        # else:
            # return x
        
class DecoderCNNPB3DCholesky(BasicDecoderV3): #use this one
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPB3DCholesky, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.z = 45
        self.r = 9
        self.phi = 16

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4

        # dropout for regularization
        # self.dropout = nn.Dropout3d(self._config.model.dropout_prob)
        
        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.n_latent_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.n_latent_nodes, 512, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(512),
                   # self.dropout,
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,3,3), (2,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,2,1), 0),
                   nn.BatchNorm3d(32),
                   # self.dropout,
                   nn.PReLU(32, 1.0),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,2,1), 0),
                   nn.BatchNorm3d(32),
                   # self.dropout,
                   nn.PReLU(32, 0.02),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   # nn.PReLU(1, 0.02),
                   nn.SiLU(1),
                   NoiseAdd(1.0),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)   #activations
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    

    
class DecoderCNNPBHD_MIRRORv1(BasicDecoderV3):
    def __init__(self, encArch = 'Large', num_output_nodes=None, **kwargs):
        self.encArch = encArch
        super(DecoderCNNPBHD_MIRRORv1, self).__init__(**kwargs)
        self._hit_smoothing_dist_mod = GumbelMod()
        self._inference_energy_activation_fct = ReLU()
        # self.device = x.device # add something like this to reduce computation time
        self._create_hierarchy_network()

    def _training_activation_fct(self, slope):
        return LeakyReLU(slope)

    def _create_hierarchy_network(self, level: int = 0):
        self.latent_nodes = self._config.model.n_latent_nodes_per_p * 4
        # change these variables for different HE decoder structures
        # FOR THE MIRROR HD, LET 3 SUBDECODERS GENERATE z1', z2', z3', 
        # THEN LAST SUBDECODER GENERATES THE ENTIRE SHOWER
        self.n_layers_per_subdec = 11
        self.layer_step = self._config.model.n_layers_per_subdec*144
         # varies depending on if last layer is > or < layer step
        self.hierarchical_lvls = 4

        # inp_layers = [self.latent_nodes + i * (self._config.model.n_latent_nodes_per_p + self.layer_step - 302) for i in range(self.hierarchical_lvls)] 
        inp_layers = self.hierarchical_lvls * [self.latent_nodes + self.layer_step]
        # inp_layers = [self.latent_nodes + i * self.layer_step for i in range(self.hierarchical_lvls)] 
        inp_layers[0] = self.latent_nodes
        out_layers = self.hierarchical_lvls * [self.layer_step]

        out_layers[self.hierarchical_lvls - 1] += (6480 - self.hierarchical_lvls * self.layer_step)
        # print(self.raw_layers)

        # Unbalanced Hierachical Decoder
        # inp_layers[0:5] = [self.latent_nodes]
        # out_layers[0:5] = [sum(out_layers[0:5])]

        # MIRROR HD, LAST DECODER GENERATES ENTIRE SHOWER
        out_layers[-1] = 6480
        self.raw_layers = [layers - self.latent_nodes for layers in inp_layers] + [6480]
        
        # Check Layers
        print("Layer Inputs: ", inp_layers)
        print("Layer Outputs: ", out_layers)
        print("Raw Layer Indices: ", self.raw_layers)

        self.moduleLayers = nn.ModuleList([])
        for i in range(len(inp_layers)):
            # self.moduleLayers.append(DecoderCNNPBv4_HEMOD(inp_layers[i], out_layers[i]))
            # self.moduleLayers.append(DecoderCNNPB3Dv1_HEMOD(inp_layers[i], out_layers[i]))
            # self.moduleLayers.append(DecoderCNNPB3Dv2_HEMOD(inp_layers[i], out_layers[i]))
            # self.moduleLayers.append(DecoderCNNPB3Dv3_HEMOD(inp_layers[i], out_layers[i]))
            self.moduleLayers.append(DecoderCNNPB3Dv4_HEMOD(inp_layers[i], out_layers[i]))

        self._create_skipcon_decoders()
        # not used
        sequential = sequentialMultiInput(*self.moduleLayers)
        return sequential

    def _create_skipcon_decoders(self):
        latent_inp = 2 * self._config.model.n_latent_nodes_per_p
        # latent_inp = self._config.model.n_latent_nodes_per_p
        recon_out = self.latent_nodes + self.layer_step
        self._subdec1 = nn.Conv3d(latent_inp, recon_out, kernel_size=1, stride=1, padding=0)
        self._subdec2 = nn.Conv3d(latent_inp, recon_out, kernel_size=1, stride=1, padding=0)
        self._subdec3 = nn.Conv3d(latent_inp, recon_out, kernel_size=1, stride=1, padding=0)
        self.subdecs = [self._subdec1, self._subdec2, self._subdec3]
    
    def forward(self, x, x0, act_fct_slope, x_raw):
        # self.sub_values = []
        x_lat = x
        self.x1, self.x2 = torch.tensor([]).to(x.device), torch.tensor([]).to(x.device) # store hits and activation tensors
        # Instead of in range(self.hierarchical_lvls) just use len(self.moduleLayers) to deal with unbalanced hierarchical decoders
        for lvl in range(len(self.moduleLayers)):
            cur_net = self.moduleLayers[lvl]
            # print("ins 2: ", x.shape)
            output_hits, output_activations = cur_net(x, x0)
            outputs = output_hits * output_activations
            z = outputs
            if lvl == len(self.moduleLayers) - 1:
                self.x1 = output_hits
                self.x2 = output_activations
            else:
                partition_ind_start = (len(self.moduleLayers) - 1 - lvl) * self._config.model.n_latent_nodes_per_p
                partition_ind_end = (len(self.moduleLayers) - lvl) * self._config.model.n_latent_nodes_per_p
                enc_z = torch.cat((x[:,0:self._config.model.n_latent_nodes_per_p], x[:,partition_ind_start:partition_ind_end]), dim=1)
                # enc_z = x[:,partition_ind_start:partition_ind_end]
                enc_z = torch.unflatten(enc_z, 1, (2 * self._config.model.n_latent_nodes_per_p, 1, 1, 1))
                # enc_z = torch.unflatten(enc_z, 1, (self._config.model.n_latent_nodes_per_p, 1, 1, 1))
                enc_z = self.subdecs[lvl](enc_z).view(enc_z.size(0), -1)
                # print(enc_z.shape)
                xz = torch.cat((x_lat, z), dim=1)
                # print(xz.shape)
                x = enc_z + xz
                # print("ins 1: ", x.shape)
        return self.x1, self.x2


class DecoderCNNPB3Dv4_HEMOD(BasicDecoderV3):
    def __init__(self, num_input_nodes, num_output_nodes, output_activation_fct=nn.Identity(), **kwargs):
        super(DecoderCNNPB3Dv4_HEMOD, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_input_nodes = num_input_nodes
        self.z = 45
        self.r = 9
        self.phi = 16
        self.hierarchal_outputs = num_output_nodes
        self.output_layers = int(self.hierarchal_outputs / 144)

        # self.n_latent_nodes = self._config.model.n_latent_nodes
        # self.n_latent_nodes = self._config.model.n_latent_nodes_per_p * 4
        
        # dropout for regularization
        # self.dropout = nn.Dropout3d(self._config.model.dropout_prob) # config object is NoneType error?
        # self.dropout = nn.Dropout3d(0.2)

        # self._node_sequence = [(2049, 800), (800, 700), (700, 600), (600, 550), (550, 500), (500, 6480)]
        self._layers =  nn.Sequential(
                   # nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),
                   nn.Unflatten(1, (self.num_input_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.num_input_nodes, 512, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(512),
                   # self.dropout,
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,3,3), (2,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(64),
                   # self.dropout,
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,2,1), 0),
                   nn.BatchNorm3d(32),
                   # self.dropout,
                   nn.PReLU(32, 1.0),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 0),
                   PeriodicConv3d(1, 1, (self.z - self.output_layers + 1, 1, 1), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 0),
                   nn.GroupNorm(1,64),
                   # self.dropout,
                   nn.SiLU(64),
                   # Residual(PreNorm(64, LinearAttention(64, cylindrical = False))),
                   LinearAttention(64, cylindrical = False),

                   PeriodicConvTranspose3d(64, 32, (5,3,3), (2,2,1), 0),
                   nn.GroupNorm(1,32),
                   # self.dropout,
                   nn.SiLU(32),
                   LinearAttention(32, cylindrical = False),

                   PeriodicConvTranspose3d(32, 1, (5,2,3), (1,1,1), 0),
                   PeriodicConv3d(1, 1, (self.z - self.output_layers + 1, 1, 1), (1,1,1), 0),
                   # nn.BatchNorm3d(45),
                   nn.SiLU(1),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0], self.hierarchal_outputs), x2.reshape(x1.shape[0], self.hierarchal_outputs)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
