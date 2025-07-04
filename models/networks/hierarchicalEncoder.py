f"""
Hierarchical Encoder

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

#logging module with handmade settings.
from CaloQVAE import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
                  
from models.networks.basicCoders import BasicEncoder
from utils.dists.distributions import SpikeAndExponentialSmoother
from utils.dists.mixtureexp import MixtureExp
from utils.dists.mixtureexpmod import MixtureExpMod
from utils.dists.gumbelmod import GumbelMod

_SMOOTHER_DICT = {"SpikeExp" : SpikeAndExponentialSmoother,
                  "MixtureExp" : MixtureExp,
                  "Gumbel" : None}
_SMOOTHER_MOD_DICT = {"MixtureExp" : MixtureExpMod,
                      "SpikeExp" : None,
                      "Gumbel" : GumbelMod}

class HierarchicalEncoder(BasicEncoder):
    def __init__(self, activation_fct=nn.ReLU(), input_dimension=784, n_latent_hierarchy_lvls=4, n_latent_nodes=100, n_encoder_layer_nodes=200, n_encoder_layers=2, skip_latent_layer=False, smoother="SpikeExp", **kwargs):

        super(HierarchicalEncoder, self).__init__(**kwargs)
        
        #TODO this assumes MNIST dataset without sequential layers
        if isinstance(input_dimension, int):
            self.num_input_nodes = input_dimension
        elif isinstance(input_dimension, list):
            self.num_input_nodes = sum(input_dimension)

        #number of hierarchy levels in encoder. This is the number of latent
        #layers. At each hiearchy level an output layer is formed.
        self.n_latent_hierarchy_lvls=n_latent_hierarchy_lvls

        #number of latent nodes in the prior - output nodes for each level of
        #the hierarchy. Also number of input nodes to the decoder, first layer
        self.n_latent_nodes=n_latent_nodes

        #each hierarchy has NN with n_encoder_layers_enc layers
        #number of deterministic nodes in each encoding layer. These layers map
        #input to the latent layer. 
        self.n_encoder_layer_nodes=n_encoder_layer_nodes
        
        # number of deterministic layers in each conditional p(z_i | z_{k<i})
        self.n_encoder_layers=n_encoder_layers

        # for all layers except latent (output)
        self.activation_fct=activation_fct

        #list of all networks in the hierarchy of the encoder
        self._networks=nn.ModuleList([])
        
        #skip_latent_layer: instead of having a single latent layer, use
        #Gaussian trick of VAE: construct mu+eps*sqrt(var) on each hierarchy
        #level. This gives n_latent_hierarchy_lvls latent variables, which
        #are then combined outside this class into one layer.
        self.skip_latent_layer=skip_latent_layer

        assert smoother in _SMOOTHER_DICT.keys(), "smoother should be one of" + str(_SMOOTHER_DICT.keys())
        self.smoothing_dist=_SMOOTHER_DICT[smoother]
        
        assert smoother in _SMOOTHER_MOD_DICT.keys(), "smoother should be one of" + str(_SMOOTHER_MOD_DICT.keys())
        self.smoothing_dist_mod = _SMOOTHER_MOD_DICT[smoother]() if _SMOOTHER_MOD_DICT[smoother] is not None else None
        
        #for each hierarchy level create a network. Input unit count will increase
        #per level.
        for lvl in range(self.n_latent_hierarchy_lvls):
            network=self._create_hierarchy_network(level=lvl)
            self._networks.append(network)

    def _create_hierarchy_network(self, level=0):       
        #skip_latent_layer: instead of having a single latent layer, use
        #Gaussian trick of VAE: construct mu+eps*sqrt(var) on each hierarchy level
        # this is done outside this class...  
        #TODO this should be revised with better structure for input layer config
        if level == 0:
            layers = [self.num_input_nodes] + list(self._config.model.encoder_hidden_nodes) + [self.n_latent_nodes]
        else:
            layers=[self.num_input_nodes+(level*self.n_latent_nodes)]+[self.n_encoder_layer_nodes]*self.n_encoder_layers+[self.n_latent_nodes]

        moduleLayers=nn.ModuleList([])
        for l in range(len(layers)-1):
            moduleLayers.append(nn.Linear(layers[l], layers[l+1]))
            #apply the activation function for all layers except the last (latent) layer 
            act_fct = nn.Identity() if l==len(layers)-2 else self.activation_fct
            moduleLayers.append(act_fct)

        sequential=nn.Sequential(*moduleLayers)
        return sequential
    
    def hierarchical_posterior(self, x, is_training=True):
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
        logger.debug("ERROR Encoder::hierarchical_posterior")
        
        posterior = []
        post_samples = []
        
        #loop hierarchy levels. apply previously defined network to input.
        #input is concatenation of data and latent variables per layer.
        for lvl in range(self.n_latent_hierarchy_lvls):
            
            #network of hierarchy level lvl 
            current_net=self._networks[lvl]
            #input to this level current_net
            current_input=torch.cat([x]+post_samples,dim=-1)
            #feed network and retrieve logit
            logits=current_net(current_input)
            
            #build the posterior distribution for this hierarchy
            #TODO this needs a switch: training smoothing, evaluation bernoulli
            posterior_dist = self.smoothing_dist(logits=logits,
                                                 beta=self._config.model.beta_smoothing_fct)
            samples=posterior_dist.reparameterise()
            posterior.append(posterior_dist)
            post_samples.append(samples)
            
        return posterior, post_samples

    def forward(self, x, is_training=True, beta_smoothing_fct=5):
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
        logger.debug("ERROR Encoder::hierarchical_posterior")
        
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
            logits=torch.clamp(current_net(current_input), min=-88., max=88.)
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

if __name__=="__main__":
    logger.debug("Testing Networks")
    logger.debug("Success")