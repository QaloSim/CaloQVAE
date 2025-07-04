"""
PCD Sampler

Author: Abhi (abhishek@myumanitoba.ca)
"""

from models.samplers.baseSampler import BaseSampler
import torch

class GS(BaseSampler):
    
    def __init__(self, batch_size, RBM, **kwargs):
        super(GS, self).__init__(**kwargs)
        assert RBM is not None
        self._RBM = RBM
        self._MCState = (torch.rand(batch_size, self._RBM.visible_bias.size(0)) >= 
                         torch.rand(batch_size, self._RBM.visible_bias.size(0))).float()
        self._batch_size = batch_size
        
    def get_batch_size(self):
        return self._batch_size
    
    @property
    def rbm(self):
        return self._RBM
    
    @rbm.setter
    def rbm(self, rbm):
        self._RBM = rbm
        
    def hidden_samples(self, visible_states):
        """
        Sample batch of hidden states given a batch of visible states
        
        Args:
            visible_states : Tensor, Dims=(batch_size * nVisibleNodes)
        Output:
            hidden_states sample : Tensor, Dims=(batch_size * nHiddenNodes)
        """
        hidden_activations = (torch.matmul(visible_states, self._RBM.weights)
                          + self._RBM.hidden_bias)
        hidden_probs = torch.sigmoid(hidden_activations)
        return (hidden_probs >= torch.rand(hidden_probs.size(), device=hidden_probs.device)).float()
        
    def visible_samples(self, hidden_states):
        """
        Sample batch of visible states given a batch of hidden states
        
        Args:
            hidden_states : Tensor, Dims=(batch_size * nHiddenNodes)
        Output:
            visible_states sample : Tensor, Dims=(batch_size * nVisibleNodes)
        """
        visible_activations = (torch.matmul(hidden_states, self._RBM.weights.t()) 
                           + self._RBM.visible_bias)
        visible_probs = torch.sigmoid(visible_activations)
        return (visible_probs >= torch.rand(visible_probs.size(), device=visible_probs.device)).float()
    
    def block_gibbs_sampling(self, post_zetas_vis=None, method='Rdm'):
        """
        Block Gibbs sampling with initialization through a persistent Markov Chain
        
        Returns:
            visible_states : Batch of visible states at end of Gibbs sampling, Dims=(batch_size * nVisibleNodes)
            hidden_states : Batch of hidden states at end of Gibbs sampling, Dims=(batch_size * nHiddenNodes)
        """
        if method == 'Rdm' or post_zetas_vis==None:
            self._MCState = (torch.rand(self._batch_size , self._RBM.visible_bias.size(0)) >= 
                         torch.rand(self._batch_size , self._RBM.visible_bias.size(0))).float()
        elif method == 'CD':
            self._MCState = post_zetas_vis
        elif method == 'PCD':
            pass
            
        visible_states = self._MCState.to(self._RBM.visible_bias.device)
        for step in range(self.n_gibbs_sampling_steps):
            hidden_states = self.hidden_samples(visible_states)
            visible_states = self.visible_samples(hidden_states)
        
        # PCD to standard Gibbs sampling
        # self._MCState = visible_states
        if method == 'PCD':
            self._MCState = visible_states
        
        return visible_states.detach(), hidden_states.detach()
