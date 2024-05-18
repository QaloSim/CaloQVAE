"""
Decoder

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.networks.networks import Network, NetworkV2, NetworkV3
from models.networks.basicCoders import BasicDecoderV3

#logging module with handmade settings.
# from CaloQVAE import logging
# logger = logging.getLogger(__name__)

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
    
    
class PeriodicConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConvTranspose2d, self).__init__()
        self.padding = padding
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Pad input tensor with periodic boundary conditions
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        # Apply convolution
        x = self.conv(x)
        return x