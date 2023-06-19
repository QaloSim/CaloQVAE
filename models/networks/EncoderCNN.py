"""
Encoder CNN

Changes the way the encoder is constructed wrt V2. No hierarchy! Everyone's equal!

"""
import torch.nn as nn  
from models.networks.hierarchicalEncoder import HierarchicalEncoder

class EncoderCNN(HierarchicalEncoder):
    def __init__(self, **kwargs):
        super(EncoderCNN, self).__init__(**kwargs)
        
    def _create_hierarchy_network(self, level: int = 0):
        """Overrides _create_hierarchy_network in HierarchicalEncoder
        :param level
        """
#         layers = [self.num_input_nodes + (level*self.n_latent_nodes)] + \
#             list(self._config.model.encoder_hidden_nodes) + \
#             [self.n_latent_nodes]

#         moduleLayers = nn.ModuleList([])
#         for l in range(len(layers)-1):
#             moduleLayers.append(nn.Linear(layers[l], layers[l+1]))
#             # apply the activation function for all layers except the last
#             # (latent) layer
#             act_fct = nn.Identity() if l==len(layers)-2 else self.activation_fct
#             moduleLayers.append(act_fct)

        sequential = nn.Sequential(
                   nn.Linear(self.num_input_nodes, 24*24),
                   nn.Unflatten(1, (1,24, 24)),
    
                   nn.Conv2d(1, 64, 3, 1, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),

                   nn.Conv2d(64, 128, 3, 1, 0),
                   nn.MaxPool2d(2,stride=2),
                   nn.PReLU(128, 0.02),

                   nn.Conv2d(128, 256, 3, 1, 0),
                   nn.PReLU(256, 0.02),

                   nn.Conv2d(256, 512, 2, 1, 0),
                   nn.PReLU(512, 0.02),
    
                   nn.Conv2d(512, 1024, 2, 1, 0),
                   nn.MaxPool2d(2,stride=2),
                   nn.PReLU(1024, 0.02),
    
                   nn.Conv2d(1024, self.n_latent_nodes, 2, 1, 0),
                   nn.MaxPool2d(2,stride=2),
                   nn.PReLU(self.n_latent_nodes, 0.02),
#                    nn.Sigmoid(),
    
                   nn.Flatten(),
                                   )
        return sequential
