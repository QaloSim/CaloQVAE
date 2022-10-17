"""
PyTorch implementation of a restricted Boltzmann machine with a Chimera-QPU topology
"""
import numpy as np
import torch
import math

from torch import nn
from torch.distributions import Distribution, Normal, Uniform

from dwave.system import DWaveSampler

from models.rbm.rbm import RBM

from CaloQVAE import logging
logger = logging.getLogger(__name__)

_CELL_SIDE_QUBITS = 4
_MAX_ROW_COLS = 16

class QimeraRBM(RBM):
    def __init__(self, n_visible, n_hidden, bernoulli=False, **kwargs):
        super(QimeraRBM, self).__init__(n_visible, n_hidden, **kwargs)

        self._n_visible=n_visible
        self._n_hidden=n_hidden
        
        require_grad=True
        
        n_cells = max(math.ceil(n_visible/_CELL_SIDE_QUBITS), math.ceil(n_hidden/_CELL_SIDE_QUBITS))
        n_rows = math.ceil(math.sqrt(n_cells))
        n_cols = n_rows
        
        assert n_cols<=_MAX_ROW_COLS
        
        visible_qubit_idxs = []
        hidden_qubit_idxs = []
        
        qpu_sampler = DWaveSampler(solver={"topology__type":"chimera", "chip_id":"DW_2000Q_6"})
        qpu_nodes = qpu_sampler.nodelist
        qpu_edges = qpu_sampler.edgelist
        
        ## The n_rows+1 and n_cols+1 ensure that we get correct number of visible and hidden qubits
        for row in range(n_rows+1):    
            for col in range(n_cols+1):
                for n in range(_CELL_SIDE_QUBITS):
                    if (len(visible_qubit_idxs) < n_visible) or (len(hidden_qubit_idxs) < n_hidden):
                        idx = 8*row + 8*col*_MAX_ROW_COLS + n
                        # Even cell
                        if (row+col)%2 == 0:
                            if idx in qpu_nodes:
                                visible_qubit_idxs.append(idx)
                            if idx+4 in qpu_nodes:
                                hidden_qubit_idxs.append(idx+4)
                        # Odd cell
                        else:
                            if idx in qpu_nodes:
                                hidden_qubit_idxs.append(idx)
                            if idx+4 in qpu_nodes:
                                visible_qubit_idxs.append(idx+4)
                                
        # Remove extra nodes from the qubit idxs lists
        visible_qubit_idxs = visible_qubit_idxs[:n_visible]
        hidden_qubit_idxs = hidden_qubit_idxs[:n_hidden]
                            
        # Prune the edgelist to remove couplings between qubits not in the RBM
        pruned_edge_list = []
        for edge in qpu_edges:
            # Coupling between RBM qubits
            if (edge[0] in visible_qubit_idxs and edge[1] in hidden_qubit_idxs) or (edge[0] in hidden_qubit_idxs and edge[1] in visible_qubit_idxs):
                pruned_edge_list.append(edge)
                
        self._visible_qubit_idxs = visible_qubit_idxs
        self._hidden_qubit_idxs = hidden_qubit_idxs
        self._pruned_edge_list = pruned_edge_list
        
        # Chimera-RBM matrix
        visible_qubit_idx_map = {visible_qubit_idx:i for i, visible_qubit_idx in enumerate(visible_qubit_idxs)}
        hidden_qubit_idx_map = {hidden_qubit_idx:i for i, hidden_qubit_idx in enumerate(hidden_qubit_idxs)}
        
        weights_mask = torch.zeros(n_visible, n_hidden, requires_grad=False)
        if not bernoulli:
            for edge in pruned_edge_list:
                if edge[0] in visible_qubit_idxs:
                    weights_mask[visible_qubit_idx_map[edge[0]], hidden_qubit_idx_map[edge[1]]] = 1.
                else:
                    weights_mask[visible_qubit_idx_map[edge[1]], hidden_qubit_idx_map[edge[0]]] = 1.
            logger.debug("weights_mask = ", weights_mask)
                        
        #arbitrarily scaled by 0.01 
        self._weights = nn.Parameter(torch.randn(n_visible, n_hidden), requires_grad=require_grad)
        #self._weights = nn.Parameter(3.*torch.rand(n_visible, n_hidden) + 1., requires_grad=require_grad)
        
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

