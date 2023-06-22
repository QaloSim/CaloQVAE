"""
Autoencoders

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch
import torch.nn as nn

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
        
#         self._layers = nn.Sequential(
#                    nn.Unflatten(1, (1001, 1,1)),
    
#                    nn.ConvTranspose2d(1001, 512, 4, 1, 0),
#                    nn.BatchNorm2d(512),
#                    nn.PReLU(512, 0.02),

#                    nn.ConvTranspose2d(512, 256, 4, 2, 0),
#                    nn.PReLU(256, 0.02),

#                    nn.ConvTranspose2d(256, 64, 4, 2, 0),
#                    nn.PReLU(64, 0.02),

#                    nn.ConvTranspose2d(64, 1, 3, 1, 0),
#                    nn.BatchNorm2d(1),
#                    nn.PReLU(1, 0.02),
    
#                    nn.Flatten(),
#                    nn.Linear(576,550),
#                                    )

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
        
#         for layer in self._layers:
#             x=self._activation_fct(layer(x))
#         x = self._layers(x)
            
#         nr_layers=len(self._layers2)
#         x1, x2 = x, x
        
#         for idx, (layer2, layer3) in enumerate(zip(self._layers2, self._layers3)):
#             if idx==nr_layers-1 and self._output_activation_fct:
#                 x1 = self._output_activation_fct(layer2(x1))
#                 x2 = self._output_activation_fct(layer3(x2))
#             else:
#                 x1 = self._activation_fct(layer2(x1))
#                 x2 = self._activation_fct(layer3(x2))
                
        x = self._layers(x)
        # x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,21,21)), 1)
        x1 = self._layers2(x)
        x2 = self._layers3(x)
        return x1, x2
    
if __name__=="__main__":
    logger.debug("Testing Networks")

    logger.debug("Success")