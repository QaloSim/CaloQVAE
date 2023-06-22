"""
Encoder CNN

Changes the way the encoder is constructed wrt V2. No hierarchy! Everyone's equal!

"""
import torch
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

        # self.sequential = nn.Sequential(
        #            nn.Linear(self.num_input_nodes, 24*24),
        #            nn.Unflatten(1, (1,24, 24)),
    
        #            nn.Conv2d(1, 64, 3, 1, 0),
        #            nn.BatchNorm2d(64),
        #            nn.PReLU(64, 0.02),

        #            nn.Conv2d(64, 128, 3, 1, 0),
        #            nn.MaxPool2d(2,stride=2),
        #            nn.PReLU(128, 0.02),

        #            nn.Conv2d(128, 256, 3, 1, 0),
        #            nn.PReLU(256, 0.02),

        #            nn.Conv2d(256, 512, 2, 1, 0),
        #            nn.PReLU(512, 0.02),
    
        #            nn.Conv2d(512, 1024, 2, 1, 0),
        #            nn.MaxPool2d(2,stride=2),
        #            nn.PReLU(1024, 0.02),
    
        #            nn.Conv2d(1024, self.n_latent_nodes, 2, 1, 0),
        #            nn.MaxPool2d(2,stride=2),
        #            nn.PReLU(self.n_latent_nodes, 0.02),
        #            # nn.Sigmoid(),
    
        #            nn.Flatten(),
        #                            )
        self.sequential = nn.Sequential(
                   nn.Linear(self.num_input_nodes, 24*24),
                   nn.Unflatten(1, (1,24, 24)),
    
                   nn.Conv2d(1, 64, 3, 1, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),
                )
        self.sequential2 = nn.Sequential(
                   nn.Conv2d(65, 128, 3, 1, 0),
                   nn.MaxPool2d(2,stride=2),
                   
                   nn.PReLU(128, 0.02),
                   nn.BatchNorm2d(128),
                   

                   nn.Conv2d(128, 256, 3, 1, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                   

                   nn.Conv2d(256, 512, 2, 1, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   
    
                   nn.Conv2d(512, 1024, 2, 1, 0),
                   nn.MaxPool2d(2,stride=2),
                   nn.BatchNorm2d(1024),
                   nn.PReLU(1024, 0.02),
                   
    
                   nn.Conv2d(1024, self.n_latent_nodes, 2, 1, 0),
                   nn.MaxPool2d(2,stride=2),
                   # nn.PReLU(self.n_latent_nodes, 0.02),
                   nn.Sigmoid(),
                   # nn.BatchNorm2d(self.n_latent_nodes),
    
                   nn.Flatten(),
                                   )

        self.sequential3 = nn.Sequential(
                   nn.Conv2d(65, 128, 3, 1, 0),
                   nn.MaxPool2d(2,stride=2),
                   
                   nn.PReLU(128, 0.02),
                   nn.BatchNorm2d(128),
                   

                   nn.Conv2d(128, 256, 3, 1, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                   

                   nn.Conv2d(256, 512, 2, 1, 0),
                   nn.BatchNorm2d(512),
                   nn.PReLU(512, 0.02),
                   
    
                   nn.Conv2d(512, 1024, 2, 1, 0),
                   nn.MaxPool2d(2,stride=2),
                   nn.BatchNorm2d(1024),
                   nn.PReLU(1024, 0.02),
                   
    
                   nn.Conv2d(1024, self.n_latent_nodes, 2, 1, 0),
                   nn.MaxPool2d(2,stride=2),
                   # nn.PReLU(self.n_latent_nodes, 0.02),
                   nn.Sigmoid(),
                   # nn.BatchNorm2d(self.n_latent_nodes),
    
                   nn.Flatten(),
                                   )
        return nn.Sequential(self.sequential, self.sequential3)

    def forward2(self, x, x0, is_training=True):
        """Overrides forward in HierarchicalEncoder
        :param level
        """
        x = self.sequential(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,22,22)), 1)
        x = self.sequential2(x)
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
        # logits=torch.clamp(current_net(current_input), min=-88., max=88.)
        logits=torch.clamp(self.forward2(current_input, x0), min=-88., max=88.)
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
        
