"""
PyTorch implementation of a quadripartite Boltzmann machine with a 
Pegasus/Advantage QPU topology
"""
import itertools
import numpy as np
import torch
from torch import nn

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class PegasusRBM(nn.Module):
    def __init__(self, n_latent_nodes, **kwargs):
        """__init__()

        Initialize an instance of a 4-partite PegasusRBM
        """
        super(PegasusRBM, self).__init__(**kwargs)
        
        # RBM constants
        self._num_partitions = 4

        assert (n_latent_nodes % self._num_partitions) == 0, \
            'Number of latent nodes should be '

        # Emtpy RBM parameters
        self._weight_dict = nn.ParameterDict()
        self._bias_dict = nn.ParameterDict()
        self._nodes_per_partition = n_latent_nodes // self._num_partitions
        
        # Dict of RBM weights for different partition combinations
        for key in itertools.combinations(range(self._num_partitions), 2):
            str_key = ''.join([str(key[i]) for i in range(len(key))])
            self._weight_dict[str_key] = nn.Parameter(torch.randn(
                self._nodes_per_partition, self._nodes_per_partition),
                requires_grad=True)

        # Dict of RBM biases for each partition
        for i in range(self._num_partitions):
            self._bias_dict[str(i)] = nn.Parameter(
                torch.randn(self._nodes_per_partition),
                requires_grad=True)

    @property
    def num_partitions(self):
        return self._num_partitions

    @property
    def nodes_per_partition(self):
        return self._nodes_per_partition

    @property
    def weight_dict(self):
        return self._weight_dict

    @property
    def bias_dict(self):
        return self._bias_dict

if __name__=="__main__":
    logger.debug("Testing PegasusRBM")
    pRBM = PegasusRBM(1024)
    print(pRBM._bias_dict.keys())
    print(pRBM._weight_dict.keys())
    logger.debug("Success")