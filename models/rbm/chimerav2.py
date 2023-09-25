"""
Chimera adapted from Pegasus architecture.
"""
import numpy as np
import torch
import math

from torch import nn
from torch.distributions import Distribution, Normal, Uniform

from dwave.system import DWaveSampler

from models.rbm.pegasusRBM import PegasusRBM

from CaloQVAE import logging
logger = logging.getLogger(__name__)


_CELL_SIDE_QUBITS = 4
_MAX_ROW_COLS = 16

class QimeraRBM(PegasusRBM):
    def __init__(self, n_visible, n_hidden, bernoulli=False, **kwargs):
        super(QimeraRBM, self).__init__(nodes_per_partition=n_visible, **kwargs)
        
        require_grad=True
        
        #arbitrarily scaled by 0.01 
        self._weights = nn.Parameter(torch.randn(n_visible, n_visible), requires_grad=require_grad)
        #self._weights = nn.Parameter(3.*torch.rand(n_visible, n_hidden) + 1., requires_grad=require_grad)
        
        weights_mask = self._weight_mask_dict['01']
        self._weights_mask = nn.Parameter(weights_mask, requires_grad=False)

        # all biases initialised to 0.5
        self._visible_bias = nn.Parameter(torch.ones(n_visible) * 0.5, requires_grad=require_grad)
        # #applying a 0 bias to the hidden nodes
        self._hidden_bias = nn.Parameter(torch.zeros(n_hidden), requires_grad=require_grad)
        
    @property
    def weights(self):
        return self._weights * self._weights_mask
    
    @weights.setter
    def weights(self, weights):
        self._weights = nn.Parameter(weights)
        
    @property
    def weights_mask(self):
        return self._weights_mask
    
    @property
    def visible_bias(self):
        return self._visible_bias
    
    @visible_bias.setter
    def visible_bias(self, v_bias):
        self._visible_bias = nn.Parameter(v_bias)
    
    @property
    def hidden_bias(self):
        return self._hidden_bias
    
    @hidden_bias.setter
    def hidden_bias(self, h_bias):
        self._hidden_bias = nn.Parameter(h_bias)
    
    @property
    def visible_qubit_idxs(self):
        return self._visible_qubit_idxs
    
    @property
    def hidden_qubit_idxs(self):
        return self._hidden_qubit_idxs
    
    @property
    def pruned_edge_list(self):
        return self._pruned_edge_list

        

if __name__=="__main__":
    logger.debug("Testing chimeraRBM")
    cRBM = QimeraRBM(8, 8)
    print(cRBM.weights)
    logger.debug("Success")

