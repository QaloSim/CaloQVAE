"""
Encoder CNN

Changes the way the encoder is constructed wrt V2.

"""
import torch
import torch.nn as nn  
from models.networks.hierarchicalEncoder import HierarchicalEncoder

class EncoderUCNN(HierarchicalEncoder):
    def __init__(self, **kwargs):
        super(EncoderUCNN, self).__init__(**kwargs)
        self.minEnergy = 256.0
        
    def _create_hierarchy_network(self, level: int = 0):
        """Overrides _create_hierarchy_network in HierarchicalEncoder
        :param level
        """
        self.seq1 = nn.Sequential(
                   nn.Linear(self.num_input_nodes, 24*24),
                   nn.Unflatten(1, (1,24, 24)),
    
                   nn.Conv2d(1, 16, 3, 1, 0),
                   nn.BatchNorm2d(16),
                   nn.PReLU(16, 0.02),
                )
        self.seq2 = nn.Sequential(
                           nn.Conv2d(17, 32, 3, 1, 0),
                           # nn.MaxPool2d(2,stride=2),
                           
                           nn.PReLU(32, 0.02),
                           nn.BatchNorm2d(32),
                        )
        
        self.seq3 = nn.Sequential(
                           nn.Conv2d(33, 64, 3, 1, 0),
                           nn.BatchNorm2d(64),
                           nn.PReLU(64, 0.02),
                        )
        
        self.seq4 = nn.Sequential(
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
                           # nn.MaxPool2d(2,stride=2),
                           # nn.PReLU(self.n_latent_nodes, 0.02),
                           nn.Sigmoid(),
                           # nn.BatchNorm2d(self.n_latent_nodes),
            
                           nn.Flatten(),
                        )
        
        
        return nn.Sequential(self.seq1, self.seq2)   #<--- this goes to an unused place

    def forward2(self, x, x0, is_training=True):
        """Overrides forward in HierarchicalEncoder
        :param level
        """
        x = self.seq1(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq2(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq3(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq4(x)
        
        # x = self.sequential(x)
        # x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,22,22).divide(self.minEnergy).log2()), 1)
        # x = self.sequential2(x)
        return x

    def forward(self, x, x0, is_training=True, beta_smoothing_fct=5):
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
        # beta = torch.tensor(self._config.model.beta_smoothing_fct,
        #                     dtype=torch.float, device=logits.device,
        #                     requires_grad=False)
        beta = torch.tensor(beta_smoothing_fct,
                            dtype=torch.float, device=logits.device,
                            requires_grad=False)
        
        samples=self.smoothing_dist_mod(logits, beta, is_training)
        
        if type(x) is tuple:
            samples = torch.bmm(samples.unsqueeze(2), x[1].unsqueeze(2)).squeeze(2)
            
        post_samples.append(samples)
            
        return beta, post_logits, post_samples
    
    
class EncoderUCNNH(HierarchicalEncoder):
    def __init__(self, encArch = 'Large', **kwargs):
        self.encArch = encArch
        super(EncoderUCNNH, self).__init__(**kwargs)
        
        
    def _create_hierarchy_network(self, level: int = 0):
        """Overrides _create_hierarchy_network in HierarchicalEncoder
        :param level
        """
        # layers = [self.num_input_nodes + (level*self.n_latent_nodes)] + \
        #     list(self._config.model.encoder_hidden_nodes) + \
        #     [self.n_latent_nodes]
        layers = [self.num_input_nodes + (level*self.n_latent_nodes)] + \
            [self.n_latent_nodes]

        moduleLayers = nn.ModuleList([])
        for l in range(len(layers)-1):
            # moduleLayers.append(nn.Linear(layers[l], layers[l+1]))
            if self.encArch == 'Large':
                moduleLayers.append(EncoderBlock(layers[l], layers[l+1]))
            elif self.encArch == 'Small':
                moduleLayers.append(EncoderBlockSmall(layers[l], layers[l+1]))
            # apply the activation function for all layers except the last
            # (latent) layer
            # act_fct = nn.Identity() if l==len(layers)-2 else self.activation_fct
            # moduleLayers.append(act_fct)

        # sequential = nn.Sequential(*moduleLayers)
        sequential = sequentialMultiInput(*moduleLayers)
        return sequential

    def forward(self, x, x0, is_training=True, beta_smoothing_fct=5):
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
        for lvl in range(self.n_latent_hierarchy_lvls):
            
            current_net=self._networks[lvl]
            if type(x) is tuple:
                current_input=torch.cat([x[0]]+post_samples,dim=1)
            else:
                current_input=torch.cat([x]+post_samples,dim=1)

            # Clamping logit values
            logits=torch.clamp(current_net(current_input, x0), min=-88., max=88.)
            # logits=torch.clamp(self.forward2(current_input, x0), min=-88., max=88.)
            post_logits.append(logits)

            # Scalar tensor - device doesn't matter but made explicit
            # beta = torch.tensor(self._config.model.beta_smoothing_fct,
            #                     dtype=torch.float, device=logits.device,
            #                     requires_grad=False)
            beta = torch.tensor(beta_smoothing_fct,
                                dtype=torch.float, device=logits.device,
                                requires_grad=False)

            samples=self.smoothing_dist_mod(logits, beta, is_training)

            if type(x) is tuple:
                samples = torch.bmm(samples.unsqueeze(2), x[1].unsqueeze(2)).squeeze(2)

            post_samples.append(samples)
            
        return beta, post_logits, post_samples
    
    
    
class EncoderBlock(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlock, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.minEnergy = 256
        
        self.seq1 = nn.Sequential(
                   nn.Linear(self.num_input_nodes, 24*24),
                   nn.Unflatten(1, (1,24, 24)),
    
                   nn.Conv2d(1, 16, 3, 1, 0),
                   nn.BatchNorm2d(16),
                   nn.PReLU(16, 0.02),
                )
        self.seq2 = nn.Sequential(
                           nn.Conv2d(17, 32, 3, 1, 0),
                           # nn.MaxPool2d(2,stride=2),
                           
                           nn.PReLU(32, 0.02),
                           nn.BatchNorm2d(32),
                        )
        
        self.seq3 = nn.Sequential(
                           nn.Conv2d(33, 64, 3, 1, 0),
                           nn.BatchNorm2d(64),
                           nn.PReLU(64, 0.02),
                        )
        
        self.seq4 = nn.Sequential(
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
                           # nn.MaxPool2d(2,stride=2),
                           # nn.PReLU(self.n_latent_nodes, 0.02),
                           nn.Sigmoid(),
                           # nn.BatchNorm2d(self.n_latent_nodes),
            
                           nn.Flatten(),
                        )
        

    def forward(self, x, x0):
        x = self.seq1(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq2(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq3(x)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = self.seq4(x)
        
        return x

class sequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
    
class EncoderBlockSmall(nn.Module):
    def __init__(self, num_input_nodes, n_latent_nodes):
        super(EncoderBlockSmall, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.n_latent_nodes = n_latent_nodes
        self.minEnergy = 256
        
        self.seq1 = nn.Sequential(
                   nn.Linear(self.num_input_nodes, 24*24),
                   nn.Unflatten(1, (1,24, 24)),
    
                   nn.Conv2d(1, 64, 3, 2, 0),
                   nn.BatchNorm2d(64),
                   nn.PReLU(64, 0.02),
    
                   nn.Conv2d(64, 256, 3, 2, 0),
                   nn.BatchNorm2d(256),
                   nn.PReLU(256, 0.02),
                )

        self.seq2 = nn.Sequential(
                           nn.Conv2d(257, 512, 3, 1, 0),
                           nn.PReLU(512, 0.02),
                           nn.BatchNorm2d(512),

                           nn.Conv2d(512, self.n_latent_nodes, 3, 1, 0),
                           nn.BatchNorm2d(self.n_latent_nodes),
                           # nn.PReLU(self.n_latent_nodes, 0.02),

                           nn.Sigmoid(),            
                           nn.Flatten(),
                        )
        

    def forward(self, x, x0):
        x = self.seq1(x)
        # x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(1000.0)), 1)
        x = self.seq2(x)
        
        return x


# class EncoderUCNNv2(HierarchicalEncoder):
#     def __init__(self, **kwargs):
#         super(EncoderUCNNv2, self).__init__(**kwargs)
#         self.minEnergy = 256.0
        
#     def _create_hierarchy_network(self, level: int = 0):
#         """Overrides _create_hierarchy_network in HierarchicalEncoder
#         :param level
#         """
#         self.seq1 = nn.Sequential(    
#                    nn.Conv2d(self.num_input_nodes, 16, (2,3), 1, 0),
#                    nn.BatchNorm2d(16),
#                    nn.PReLU(16, 0.02),
#                 )
#         self.seq2 = nn.Sequential(
#                            nn.Conv2d(17, 32, (2,3), 1, 0),
#                            # nn.MaxPool2d(2,stride=2),
                           
#                            nn.PReLU(32, 0.02),
#                            nn.BatchNorm2d(32),
#                         )
                
#         self.seq3 = nn.Sequential(
#                            nn.Conv2d(33, 64, (2,3), 1, 0),
#                            nn.BatchNorm2d(64),
#                            nn.PReLU(64, 0.02),
#                         )
                
#         self.seq4 = nn.Sequential(
#                            nn.Conv2d(65, 128, (2,3), 1, 0),
#                            nn.MaxPool2d(2,stride=2),
                           
#                            nn.PReLU(128, 0.02),
#                            nn.BatchNorm2d(128),
                           
        
#                            nn.Conv2d(128, 256, (2,3), 1, 0),
#                            nn.BatchNorm2d(256),
#                            nn.PReLU(256, 0.02),
                           
        
#                            nn.Conv2d(256, 512, (1,2), 1, 0),
#                            nn.BatchNorm2d(512),
#                            nn.PReLU(512, 0.02),
                           
            
#                            nn.Conv2d(512, 1000, (1,2), 1, 0),
#                            nn.MaxPool2d(2,stride=2),
#                            nn.BatchNorm2d(1000),
#                            nn.Sigmoid(),
            
#                            nn.Flatten(),
#                         )
                
        
#         return nn.Sequential(self.seq1, self.seq2)   #<--- this goes to an unused place

#     def forward(self, x, x0, is_training=True):
#         """Overrides forward in HierarchicalEncoder
#         :param level
#         """
#         x = self.seq1(x)
#         x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
#         x = self.seq2(x)
#         x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
#         x = self.seq3(x)
#         x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item()).divide(self.minEnergy).log2()), 1)
#         x = self.seq4(x)
        
#         # x = self.sequential(x)
#         # x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).repeat(1,1,22,22).divide(self.minEnergy).log2()), 1)
#         # x = self.sequential2(x)
#         return x

#     def forwarddisabled(self, x, x0, is_training=True):
#         """ This function defines a hierarchical approximate posterior distribution. The length of the output is equal 
#             to n_latent_hierarchy_lvls and each element in the list is a DistUtil object containing posterior distribution 
#             for the group of latent nodes in each hierarchy level. 

#         Args:
#             input: a tensor containing input tensor.
#             is_training: A boolean indicating whether we are building a training graph or evaluation graph.

#         Returns:
#             posterior: a list of DistUtil objects containing posterior parameters.
#             post_samples: A list of samples from all the levels in the hierarchy, i.e. q(z_k| z_{0<i<k}, x).
#         """
#         # logger.debug("ERROR Encoder::hierarchical_posterior")
        
#         post_samples = []
#         post_logits = []
        
#         #loop hierarchy levels. apply previously defined network to input.
#         #input is concatenation of data and latent variables per layer.
#         # for lvl in range(self.n_latent_hierarchy_lvls):
            
#         # current_net=self._networks[0]
#         if type(x) is tuple:
#             current_input=torch.cat([x[0]]+post_samples,dim=1)
#         else:
#             current_input=torch.cat([x]+post_samples,dim=1)

#         # Clamping logit values
#         # logits=torch.clamp(current_net(current_input), min=-88., max=88.)
#         logits=torch.clamp(self.forward2(current_input, x0), min=-88., max=88.)
#         post_logits.append(logits)
        
#         # Scalar tensor - device doesn't matter but made explicit
#         beta = torch.tensor(self._config.model.beta_smoothing_fct,
#                             dtype=torch.float, device=logits.device,
#                             requires_grad=False)
        
#         samples=self.smoothing_dist_mod(logits, beta, is_training)
        
#         if type(x) is tuple:
#             samples = torch.bmm(samples.unsqueeze(2), x[1].unsqueeze(2)).squeeze(2)
            
#         post_samples.append(samples)
            
#         return beta, post_logits, post_samples
        
