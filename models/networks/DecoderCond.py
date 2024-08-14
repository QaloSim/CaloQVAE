"""
Decoder

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU, ReLU

# from models.networks.networks import Network, NetworkV2, NetworkV3
from models.networks.basicCoders import BasicDecoderV3

# get the hits and activation functions for the hierarchial decoder
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
        # self.n_latent_nodes = 302 * 4
        
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
        # Pad input tensor with periodic boundary conditions
        # if self.padding == 1:
        #     mid = x.shape[-1] // 2
        #     shift = torch.cat((x[..., [-1], mid:], x[..., [-1], :mid]), -1)
        #     x = torch.cat((x, shift), dim=-2)
        x = F.pad(x, (self.padding, self.padding, 0, 0, 0, 0), mode='circular')
        # Apply convolution
        x = self.conv(x)
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

class DecoderCNNPB3Dv2(BasicDecoderV3):
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