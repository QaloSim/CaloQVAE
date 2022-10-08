"""
PyTorch implementation of a quadripartite Boltzmann machine with a 
Pegasus/Advantage QPU topology
"""
import itertools
import torch
from torch import nn

class PegasusRBM(nn.Module):
    def __init__(self, nodes_per_layer: int, **kwargs):
        """__init__()

        Initialize an instance of a 4-partite PegasusRBM
        """
        super(PegasusRBM, self).__init__(**kwargs)

        # RBM constants
        self._num_partitions = 4

        # Emtpy RBM parameters
        self._weight_dict = nn.ParameterDict()
        self._bias_dict = nn.ParameterDict()
        self._nodes_per_partition = nodes_per_layer

        # Dict of RBM weights for different partition combinations
        for key in itertools.combinations(range(self._num_partitions), 2):
            str_key = ''.join([str(key[i]) for i in range(len(key))])
            self._weight_dict[str_key] = nn.Parameter(torch.randn((
                self._nodes_per_partition, self._nodes_per_partition),
                requires_grad=True))

        # Dict of RBM biases for each partition
        for i in range(self._num_partitions):
            self._bias_dict[str(i)] = nn.Parameter(
                torch.randn(self._nodes_per_partition),
                requires_grad=True)

    @property
    def nodes_per_partition(self):
        """Accessor method for a protected variable

        :return: no. of latent nodes per partition
        """
        return self._nodes_per_partition

    @property
    def weight_dict(self):
        """Accessor method for a protected variable

        :return: dict with partition combinations as str keys ('01', '02', ...)
                 and partition weight matrices as values (w_01, w_02, ...) 
        """
        return self._weight_dict

    @property
    def bias_dict(self):
        """Accessor method for a protected variable

        :return: dict with partition no.s as str keys ('0', '1', ...)
                 and partition biases as values (b_0, b_1, ...)
        """
        return self._bias_dict


if __name__ == "__main__":
    pRBM = PegasusRBM(1024)
    print(pRBM.bias_dict.keys())
    print(pRBM.weight_dict.keys())
