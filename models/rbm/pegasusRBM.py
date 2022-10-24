"""
PyTorch implementation of a quadripartite Boltzmann machine with a 
Pegasus/Advantage QPU topology
"""
import itertools
import math

import dwave_networkx as dnx
import torch
from torch import nn
from dwave.system import DWaveSampler

_MAX_ROW_COLS = 16

class PegasusRBM(nn.Module):
    """
    PyTorch implementation of a quadripartite Boltzmann machine with a 
    Pegasus/Advantage QPU topology
    """
    def __init__(self, nodes_per_layer: int, qpu: bool = True, **kwargs):
        """__init__()

        Initialize an instance of a 4-partite PegasusRBM

        :param nodes_per_layer (int) : Number of nodes for each partition
        :param qpu (bool) : Only allow connections present on the QPU 
        """
        super(PegasusRBM, self).__init__(**kwargs)

        # RBM constants
        self._n_partitions = 4

        # Emtpy RBM parameters
        self._weight_dict = nn.ParameterDict()
        self._bias_dict = nn.ParameterDict()
        self._nodes_per_partition = nodes_per_layer

        # Dict of RBM weights for different partition combinations
        for key in itertools.combinations(range(self._n_partitions), 2):
            str_key = ''.join([str(key[i]) for i in range(len(key))])
            self._weight_dict[str_key] = nn.Parameter(
                torch.randn(self._nodes_per_partition,
                            self._nodes_per_partition), requires_grad=True)

        # Dict of RBM biases for each partition
        for i in range(self._n_partitions):
            self._bias_dict[str(i)] = nn.Parameter(
                torch.randn(self._nodes_per_partition), requires_grad=True)
            
        if qpu:
            self._qubit_idx_dict = self.generate_qubit_idx_dict()



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

    def generate_qubit_idx_dict(self):
        """Partition the qubits on the device into 4-partite BM

        :return: dict with partition no.s as str keys ('0', '1', ...)
                 and list of qubit idxs for each partition as values
        """
        # Coordinate system to convert 'nice' coordinates into linear
        coords = dnx.pegasus_coordinates(16)

        # Initialize the dict to be returned
        qubit_idx_dict = {}
        for partition in range(self._n_partitions):
            qubit_idx_dict[str(partition)] = []
    
        n_nodes = self._nodes_per_partition * self._n_partitions
        n_cells = math.ceil(n_nodes/24)
        n_rows = math.ceil(math.sqrt(n_cells))
        n_cols = n_rows

        device = DWaveSampler(solver={'chip_id': 'Advantage_system_4.1'})
        qpu_nodes = device.nodelist
        qpu_edges = device.edgelist

        for z, row, col in itertools.product(range(3), , range(2)):
            

        return qubit_idx_dict

if __name__ == "__main__":
    pRBM = PegasusRBM(1024)
    print(pRBM.bias_dict.keys())
    print(pRBM.weight_dict.keys())
