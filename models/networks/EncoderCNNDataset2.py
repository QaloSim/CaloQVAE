"""
Autoencoders

Author: Soren Andersen (sandersen@triumf.ca)

"""

import torch
import torch.nn as nn  
from models.networks.hierarchicalEncoder import HierarchicalEncoder

class EncoderCNN_D2(HierarchicalEncoder):
    def __init__(self, **kwargs):
        super(EncoderCNN_D2, self).__init__(**kwargs)
    """Overrides _create_hierarchy_network in HierarchicalEncoder
        :param level
        """
    def _create_hierarchy_network(self, level: int = 0):
        self.sequential = nn.Sequential(
                   nn.Linear(self.num_input_nodes, 88*88),
                   nn.Unflatten(1, (1,88,88)),

                   nn.Conv2d(1, 16, 3, 1, 0),             #<--- used to be 16 to 64
                   nn.BatchNorm2d(16),
                   nn.PReLU(16, 0.02),

                   nn.Conv2d(16, 128, 3, 1, 0),
                   nn.MaxPool2d(2,stride=2),

                   nn.PReLU(128, 0.02),
                   nn.BatchNorm2d(128),

                   nn.Conv2d(128, 256, 3, 1, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),


                   nn.Conv2d(256, 512, 3, 1, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),


                   nn.Conv2d(512, 1024, 3, 1, 0),
                   nn.MaxPool2d(2,stride=2),
                   nn.BatchNorm2d(1024),
                   nn.PReLU(1024, 0.02),


                   nn.Conv2d(1024, self.n_latent_nodes, 3, 1, 0),
                   nn.MaxPool2d(2,stride=2),
                   # nn.PReLU(self.n_latent_nodes, 0.02),
                   nn.Sigmoid(),
                   # nn.BatchNorm2d(self.n_latent_nodes),

                   nn.Flatten(),
                                   )

        return nn.Sequential(self.sequential)  #<--- goes to an unused place
    
    def forward2(self, x, is_training=True):
            """Overrides forward in HierarchicalEncoder
            :param level
            """
            x = self.sequential(x)
            return x

    def forward(self, x, x0, is_training=True):
        """ This function defines a hierarchical approximate posterior distribution. The length of the output is equal 
            to n_latent_hierarchy_lvls and each element in the list is a DistUtil object containing posterior distribution 
            for the group of latent nodes in each hierarchy level. 

        Args:
            input: a tensor containing input tensor.
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.

        Returns:
            posterior: a list of DistUtil objects containing posterior parameters.
            post_samples: A list of samples from all the levels in the hierarchy, i.e. q(z_k| z_{0<i<k}, x).
        """
        # logger.debug("ERROR Encoder::hierarchical_posterior")
        
        post_samples = []
        post_logits = []
        
        #loop hierarchy levels. apply previously defined network to input.
        #input is concatenation of data and latent variables per layer.
        # for lvl in range(self.n_latent_hierarchy_lvls):
            
        # current_net=self._networks[0]
        if type(x) is tuple:
            current_input=torch.cat([x[0]]+post_samples,dim=1)
        else:
            current_input=torch.cat([x]+post_samples,dim=1)

        # Clamping logit values
        logits=torch.clamp(self.forward2(current_input), min=-88., max=88.)
        #logits=torch.clamp(self.forward2(current_input, x0), min=-88., max=88.)
        post_logits.append(logits)
        
        # Scalar tensor - device doesn't matter but made explicit
        beta = torch.tensor(self._config.model.beta_smoothing_fct,
                            dtype=torch.float, device=logits.device,
                            requires_grad=False)
        
        samples=self.smoothing_dist_mod(logits, beta, is_training)
        
        if type(x) is tuple:
            samples = torch.bmm(samples.unsqueeze(2), x[1].unsqueeze(2)).squeeze(2)
            
        post_samples.append(samples)
            
        return beta, post_logits, post_samples