"""
Autoencoders

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.networks import Network, NetworkV2, NetworkV3

#logging module with handmade settings.
from CaloQVAE import logging
logger = logging.getLogger(__name__)

class BasicEncoder(Network):
    def __init__(self,**kwargs):
        super(BasicEncoder, self).__init__(**kwargs)

    def forward(self, x):
        logger.debug("Encoder::encode")
        for layer in self._layers:
            if self._activation_fct:
                x=self._activation_fct(layer(x))
            else:
                x=layer(x)
        return x

class BasicDecoder(Network):
    def __init__(self,output_activation_fct=nn.Identity(),**kwargs):
        super(BasicDecoder, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct

    def forward(self, x):
        logger.debug("Decoder::decode")
        nr_layers=len(self._layers)
        for idx,layer in enumerate(self._layers):
            if idx==nr_layers-1 and self._output_activation_fct:
                x=self._output_activation_fct(layer(x))
            else:
                x=self._activation_fct(layer(x))
        return x
    

class BasicDecoderV2(NetworkV2):
    def __init__(self, output_activation_fct=nn.Identity(),**kwargs):
        super(BasicDecoderV2, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct

    def forward(self, x):
        logger.debug("Decoder::decode")
        nr_layers=len(self._layers)
        x1, x2 = x, x
        for idx, (layer1, layer2) in enumerate(zip(self._layers, self._layers2)):
            if idx==nr_layers-1 and self._output_activation_fct:
                x1 = self._output_activation_fct(layer1(x1))
                x2 = self._output_activation_fct(layer2(x2))
            else:
                x1 = self._activation_fct(layer1(x1))
                x2 = self._activation_fct(layer2(x2))
        return x1, x2
    
class BasicDecoderV3(NetworkV3):
    def __init__(self, output_activation_fct=nn.Identity(), **kwargs):
        super(BasicDecoderV3, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        
    def forward(self, x):
        logger.debug("Decoder::decode")
        
        for layer in self._layers:
            x=self._activation_fct(layer(x))
            
        nr_layers=len(self._layers2)
        x1, x2 = x, x
        
        for idx, (layer2, layer3) in enumerate(zip(self._layers2, self._layers3)):
            if idx==nr_layers-1 and self._output_activation_fct:
                x1 = self._output_activation_fct(layer2(x1))
                x2 = self._output_activation_fct(layer3(x2))
            else:
                x1 = self._activation_fct(layer2(x1))
                x2 = self._activation_fct(layer3(x2))
        return x1, x2
    
class DecoderCNN(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNN, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.minEnergy = 256.0

        self._layers = nn.Sequential(
                   nn.Unflatten(1, (1000, 1,1)),
    
                   nn.ConvTranspose2d(1000, 512, 4, 1, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   

                   nn.ConvTranspose2d(512, 256, 4, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),

                   nn.ConvTranspose2d(256, 128, 3, 2, 0),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),
                   
                                   )
        self._layers2 = nn.Sequential(
                   nn.ConvTranspose2d(128, 32, 2, 1, 0),
                   nn.BatchNorm2d(32),
                   nn.PReLU(32, 0.02),

                   nn.ConvTranspose2d(32, 16, 2, 1, 0),
                   nn.BatchNorm2d(16),
                   nn.PReLU(16, 0.02),

                   nn.ConvTranspose2d(16, 1, 2, 1, 0),
                   nn.Dropout(0.2),                  
    
                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
#                    nn.Sigmoid(),
                                   )
        self._layers3 = nn.Sequential(
                   nn.ConvTranspose2d(128, 32, 2, 1, 0),
                   nn.BatchNorm2d(32),
                   nn.PReLU(32, 0.02),

                   nn.ConvTranspose2d(32, 16, 2, 1, 0),
                   nn.BatchNorm2d(16),
                   nn.PReLU(16, 0.02),

                   nn.ConvTranspose2d(16, 1, 2, 1, 0),
                   nn.Dropout(0.2),                  
    
                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
    def forward(self, x, x0):
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        # x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,21,21)), 1)
        # x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,21,21).divide(self.minEnergy).log2()), 1)
        x1 = self._layers2(x)
        x2 = self._layers3(x)
        return x1, x2
    
    
class DecoderCNNCond(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNCond, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.minEnergy = 256.0
        self.n_latent_nodes = self._config.model.n_latent_nodes

        self._layers = nn.Sequential(
                   # nn.Unflatten(1, (self.n_latent_nodes, 1,1)),
                   nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),

    
                   # nn.ConvTranspose2d(self.n_latent_nodes, 512, 4, 1, 0),
                   nn.ConvTranspose2d(self._node_sequence[0][0]-1, 512, 4, 1, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   

                   nn.ConvTranspose2d(512, 256, 4, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),

                   nn.ConvTranspose2d(256, 128, 3, 2, 0),
                   nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),
                   
                                   )
        self._layers2 = nn.Sequential(
                   nn.ConvTranspose2d(129, 32, 2, 1, 0),
                   nn.BatchNorm2d(32),
                   nn.PReLU(32, 0.02),

                   nn.ConvTranspose2d(32, 16, 2, 1, 0),
                   nn.BatchNorm2d(16),
                   nn.PReLU(16, 0.02),

                   nn.ConvTranspose2d(16, 1, 2, 1, 0),
                   nn.Dropout(0.2),                  
    
                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
#                    nn.Sigmoid(),
                                   )
        self._layers3 = nn.Sequential(
                   nn.ConvTranspose2d(129, 32, 2, 1, 0),
                   nn.BatchNorm2d(32),
                   nn.PReLU(32, 0.02),

                   nn.ConvTranspose2d(32, 16, 2, 1, 0),
                   nn.BatchNorm2d(16),
                   nn.PReLU(16, 0.02),

                   nn.ConvTranspose2d(16, 1, 2, 1, 0),
                   nn.Dropout(0.2),                  
    
                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
    def forward(self, x, x0):
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,21,21)), 1)
        # x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,21,21).divide(self.minEnergy).log2()), 1)
        x1 = self._layers2(x)
        x2 = self._layers3(x)
        return x1, x2
    
    
class DecoderCNNCondSmall(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNCondSmall, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.minEnergy = 256.0
        self.n_latent_nodes = self._config.model.n_latent_nodes

        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),

                   nn.ConvTranspose2d(self._node_sequence[0][0]-1, 512, 4, 2, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   

                   nn.ConvTranspose2d(512, 256, 4, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   nn.ConvTranspose2d(257, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
        self._layers3 = nn.Sequential(
                   nn.ConvTranspose2d(257, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
    def forward(self, x, x0):
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000)), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1, x2
    
    
class DecoderCNNUnconditioned(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNUnconditioned, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.minEnergy = 256.0
        self.n_latent_nodes = self._config.model.n_latent_nodes

        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),

                   nn.ConvTranspose2d(self._node_sequence[0][0]-1, 512, 4, 2, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   

                   nn.ConvTranspose2d(512, 256, 4, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   nn.ConvTranspose2d(256, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
        self._layers3 = nn.Sequential(
                   nn.ConvTranspose2d(256, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
    def forward(self, x, x0):
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        # xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000)), 1)
        x1 = self._layers2(x)
        x2 = self._layers3(x)
        return x1, x2
    
    
class DecoderCNNPosCondSmall(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNPosCondSmall, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.minEnergy = 256.0
        self.n_latent_nodes = self._config.model.n_latent_nodes

        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),

                   nn.ConvTranspose2d(self._node_sequence[0][0]-1, 512, 4, 2, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   

                   nn.ConvTranspose2d(512, 256, 4, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   nn.ConvTranspose2d(256, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
        self._layers3 = nn.Sequential(
                   nn.ConvTranspose2d(256, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
    def forward(self, x, x0):
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        xx0 = x0.unsqueeze(2).unsqueeze(3).repeat(1, x.size(1), x.size(2), x.size(3)).divide(1000.0) + x
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1, x2

    
class DecoderCNNUnconditionedAct(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNUnconditionedAct, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.minEnergy = 256.0
        self.n_latent_nodes = self._config.model.n_latent_nodes

        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),

                   nn.ConvTranspose2d(self._node_sequence[0][0]-1, 512, 4, 2, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   

                   nn.ConvTranspose2d(512, 256, 4, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   nn.ConvTranspose2d(257, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
        self._layers3 = nn.Sequential(
                   nn.ConvTranspose2d(256, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
    def forward(self, x, x0):
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000)), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(x)
        return x1, x2
    
    
    
class DecoderCNNHitsToAct(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNHitsToAct, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.minEnergy = 256.0
        self.n_latent_nodes = self._config.model.n_latent_nodes

        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),

                   nn.ConvTranspose2d(self._node_sequence[0][0]-1, 512, 4, 2, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   

                   nn.ConvTranspose2d(512, 256, 4, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   nn.ConvTranspose2d(257, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),
                    )

        self._layers2_2 = nn.Sequential(nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
        self._layers3 = nn.Sequential(
                   nn.ConvTranspose2d(256, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),
                    )
        self._layers3_2 = nn.Sequential( 
                   nn.ConvTranspose2d(2, 1, 3, 1, 0),
                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
    def forward(self, x, x0):
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000)), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(x)
        x2 = torch.cat((x2,x1), 1)
        x1 = self._layers2_2(x1)
        x2 = self._layers3_2(x2)

        return x1, x2
    
    
class DecoderCNN_nth_da_charm(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNN_nth_da_charm, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.minEnergy = 256.0
        self.n_latent_nodes = self._config.model.n_latent_nodes

        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),

                   nn.ConvTranspose2d(self._node_sequence[0][0]-1, 512, 4, 2, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   

                   nn.ConvTranspose2d(512, 256, 4, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   nn.ConvTranspose2d(257, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
        self._layers3 = nn.Sequential(
                   nn.ConvTranspose2d(257, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
    def forward(self, x, x0):
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        x_act = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000)), 1)
        x_hits = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000).multiply(-0.004).exp()), 1)
        x_hits = self._layers2(x_hits)
        x_act = self._layers3(x_act)
        return x_hits, x_act


class DecoderCNNUnconditionedHits(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(DecoderCNNUnconditionedHits, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.minEnergy = 256.0
        self.n_latent_nodes = self._config.model.n_latent_nodes

        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (self._node_sequence[0][0]-1, 1,1)),

                   nn.ConvTranspose2d(self._node_sequence[0][0]-1, 512, 4, 2, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   

                   nn.ConvTranspose2d(512, 256, 4, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   nn.ConvTranspose2d(256, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
        self._layers3 = nn.Sequential(
                   nn.ConvTranspose2d(257, 64, 4, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 1, 3, 1, 0),
                   nn.BatchNorm2d(1),
                   nn.PReLU(1, 0.02),

                   nn.Flatten(),
                   nn.Linear(576,self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                                   )
        
    def forward(self, x, x0):
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000)), 1)
        x1 = self._layers2(x)
        x2 = self._layers3(xx0)
        return x1, x2   
    
    
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
        logger.debug("Decoder::decode")
                
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
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        # xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000)), 1)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0):
        return (torch.log(x0) - log_e_min)/(log_e_max - log_e_min)
    
    
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
        logger.debug("Decoder::decode")
                
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
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000)), 1)
        x1 = self._layers2(xx0)
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0):
        return (torch.log(x0) - log_e_min)/(log_e_max - log_e_min)
    
##################################
class Classifier(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=368, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.minEnergy = 256.0

        self._layers = nn.Sequential(
                   ## nn.BatchNorm1d(self.num_output_nodes),
                   nn.LeakyReLU(0.02),
                   nn.Linear(self.num_output_nodes, 100),
                   ## nn.BatchNorm1d(100),
                   nn.LeakyReLU(0.2),
                   nn.Linear(100, 15),
                   ## nn.BatchNorm1d(15),
                                   )
        
    def forward(self, x):
        logger.debug("Classifier::classify")
                
        x = self._layers(x)
        return x


class DecoderCNNv2(BasicDecoderV3):
    def __init__(self, output_activation_fct=nn.Identity(),num_output_nodes=7, **kwargs):
        super(DecoderCNNv2, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        self.num_output_nodes = num_output_nodes
        self.minEnergy = 256.0
        
        self._layers = nn.Sequential(
                   nn.Unflatten(1, (1000, 1,1)),
    
                   nn.ConvTranspose2d(1000, 512, (2,3), 1, 0),
                   # nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   

                   nn.ConvTranspose2d(512, 256, (2,3), 1, 0),
                   # nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),

                   nn.ConvTranspose2d(256, 128, (2,3), 1, 0),
                   # nn.BatchNorm2d(128),
                   nn.PReLU(128, 0.02),
                   
                                   )
        self._layers2 = nn.Sequential(
                   nn.ConvTranspose2d(128, 64, (2,3), 1, 0),
                   # nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 32, (2,3), 1, 0),
                   # nn.BatchNorm2d(32),
                   nn.PReLU(32, 0.02),

                   nn.ConvTranspose2d(32, 16, (2,3), 1, 0),
                   # nn.BatchNorm2d(16),
                   nn.PReLU(16, 0.02), 

                   nn.ConvTranspose2d(16, 8, (2,3), 1, 0),
                   # nn.BatchNorm2d(8),
                   nn.PReLU(8, 0.02), 

                   nn.ConvTranspose2d(8, self.num_output_nodes, (2,3), 1, 0),
                   # nn.BatchNorm2d(self.num_output_nodes),
                   nn.PReLU(self.num_output_nodes, 0.02),

                   nn.ConvTranspose2d(self.num_output_nodes, self.num_output_nodes, (2,4), 1, 0),
                   # nn.BatchNorm2d(self.num_output_nodes),
                   nn.PReLU(self.num_output_nodes, 0.02),
                                   )
        
        self._layers3 = nn.Sequential(
                   nn.ConvTranspose2d(128, 64, (2,3), 1, 0),
                   # nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.ConvTranspose2d(64, 32, (2,3), 1, 0),
                   # nn.BatchNorm2d(32),
                   nn.PReLU(32, 0.02),

                   nn.ConvTranspose2d(32, 16, (2,3), 1, 0),
                   # nn.BatchNorm2d(16),
                   nn.PReLU(16, 0.02), 

                   nn.ConvTranspose2d(16, 8, (2,3), 1, 0),
                   # nn.BatchNorm2d(8),
                   nn.PReLU(8, 0.02), 

                   nn.ConvTranspose2d(8, self.num_output_nodes, (2,3), 1, 0),
                   # nn.BatchNorm2d(self.num_output_nodes),
                   nn.PReLU(self.num_output_nodes, 0.02),

                   nn.ConvTranspose2d(self.num_output_nodes, self.num_output_nodes, (2,4), 1, 0),
                   # nn.BatchNorm2d(self.num_output_nodes),
                   nn.PReLU(self.num_output_nodes, 0.02),
                                   )
        
    def forward(self, x, x0):
        logger.debug("Decoder::decode")
                
        x = self._layers(x)
        # x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,21,21)), 1)
        # x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,21,21).divide(self.minEnergy).log2()), 1)
        x1 = self._layers2(x)
        x2 = self._layers3(x)
        return x1, x2
    
if __name__=="__main__":
    logger.debug("Testing Networks")

    logger.debug("Success")