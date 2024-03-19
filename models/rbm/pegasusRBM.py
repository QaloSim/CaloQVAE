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
_CELL_SIDE_QUBITS = 4
_QPU_DEPTH = 3
_QPU_PSIZE = 16

class PegasusRBM(nn.Module):
    """
    PyTorch implementation of a quadripartite Boltzmann machine with a 
    Pegasus/Advantage QPU topology
    """
    def __init__(self, nodes_per_partition: int, qpu: bool = True, **kwargs):
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
        self._nodes_per_partition = nodes_per_partition
        self._weight_mask_dict = nn.ParameterDict()

        # Fully-connected or Pegasus-restricted 4-partite BM
        self._qpu = qpu

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
            self._qubit_idx_dict, device = self.gen_qubit_idx_dict()
            self._weight_mask_dict = self.gen_weight_mask_dict(
                self._qubit_idx_dict, device)

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
        if self._qpu:
            for key in self._weight_dict.keys():
                self._weight_dict[key] = self._weight_dict[key] \
                    * self._weight_mask_dict[key]
        return self._weight_dict


    @property
    def bias_dict(self):
        """Accessor method for a protected variable

        :return: dict with partition no.s as str keys ('0', '1', ...)
                 and partition biases as values (b_0, b_1, ...)
        """
        return self._bias_dict

    def gen_qubit_idx_dict(self):
        """Partition the qubits on the device into 4-partite BM

        :return: dict with partition no.s as str keys ('0', '1', ...)
                 and list of qubit idxs for each partition as values
        """
        # Coordinate system to convert 'nice' coordinates into linear
        coords = dnx.pegasus_coordinates(_QPU_PSIZE)

        # Initialize the dict to be returned
        idx_dict = {}
        for partition in range(self._n_partitions):
            idx_dict[str(partition)] = []

        n_nodes = self._nodes_per_partition * self._n_partitions
        n_cells = math.ceil(n_nodes/(2*_CELL_SIDE_QUBITS))
        n_cells_per_layer = math.ceil(n_cells/_QPU_DEPTH)
        n_rows = math.ceil(math.sqrt(n_cells_per_layer))
        n_cols = n_rows

        device = DWaveSampler(solver={'topology__type': 'pegasus',  "chip_id":"Advantage_system6.4"})
        # device = DWaveSampler(solver={'topology__type': 'pegasus', "chip_id":"Advantage_system4.1"})
        self._qpu_sampler = device
        qpu_nodes = device.nodelist

        # Add one extra row and col to each layer to account for dead qubits
        for row, col, layer in itertools.product(range(n_rows+1),
                                                 range(n_cols),
                                                 range(_QPU_DEPTH)):

            idx_map = {}
            # Horizontal qubits
            if row % 2 == 0:
                idx_map.update({'0': [0, 3], '2': [1, 2]})
            else:
                idx_map.update({'0': [1, 2], '2': [0, 3]})

            # Vertical qubits
            if col % 2 == 0:
                idx_map.update({'1': [0, 3], '3': [1, 2]})
            else:
                idx_map.update({'1': [1, 2], '3': [0, 3]})

            for partition, idxs in idx_map.items():
                k_side = 1 if int(partition) % 2 == 0 else 0
                for side_idx in idxs:
                    q_idx = coords.nice_to_linear((layer, col, row,
                                                   k_side, side_idx))
                    if q_idx in qpu_nodes:
                        idx_dict[partition].append(q_idx)

            curr_len_list = list(len(partition) for partition in
                                 idx_dict.values())
            if curr_len_list > [self._nodes_per_partition]*len(curr_len_list):
                break

        for partition, idxs in idx_dict.items():
            idx_dict[partition] = idx_dict[partition][:
                self._nodes_per_partition]

        # Check if any edges exist b/w nodes in a given partition
        """for partition, node_list in idx_dict.items():
            print(f"Checking partition: {partition}")
            print(len(list(itertools.combinations(node_list, 2))))
            for edge in itertools.combinations(node_list, 2):
                if edge in qpu_edges or edge[::-1] in qpu_edges:
                    edge_0 = coords.linear_to_nice(edge[0])
                    edge_1 = coords.linear_to_nice(edge[1])
                    print(f"{edge} in qpu_edges : {edge[0]} = {edge_0}",
                          f"{edge[1]} = {edge_1}")"""
        self.idx_dict = idx_dict
        return idx_dict, device

    def gen_weight_mask_dict(self, qubit_idx_dict, device):
        """Generate the weight mask for each partition-pair

        :param qubit_idx_dict (dict): Dict with partition no.s as keys and
        list of qubit idxs for each partition as values
        ;param device (DWaveSampler): QPU device containing list of nodes and 
        edges

        :return weight_mask_dict (dict): Dict with partition combinations as
        keys and weight mask for each combination as values
        """
        # Build the pruned edge list
        # List of all qubits in the BM
        idx_list = []
        for p_idx_list in qubit_idx_dict.values():
            idx_list.extend(p_idx_list)

        pruned_edge_list = []
        for edge in device.edgelist:
            if edge[0] in idx_list and edge[1] in idx_list:
                pruned_edge_list.append(edge)
                
        # print(len(device.edgelist), len(pruned_edge_list))
        self._pruned_edge_list = pruned_edge_list

        # Initialize the dict with torch.zeros
        weight_mask_dict = {}
        for key in self._weight_dict.keys():
            weight_mask_dict[key] = torch.zeros(self._nodes_per_partition,
                                                self._nodes_per_partition,
                                                requires_grad=False)

        # Add valid edges to the RBM
        for edge in pruned_edge_list:
            # Find the partitions of the nodes this edge connects
            edge_0_idx = idx_list.index(edge[0])
            edge_1_idx = idx_list.index(edge[1])

            p_0 = math.floor(edge_0_idx / self._nodes_per_partition)
            p_1 = math.floor(edge_1_idx / self._nodes_per_partition)

            # Find the idxs of the weight_mask matrix to set to 1.
            p_0_idx = edge_0_idx % self._nodes_per_partition
            p_1_idx = edge_1_idx % self._nodes_per_partition

            # Use the lower partition to index the weight_mask_dict and
            # set the matrix index accordingly
            if p_0 < p_1:
                p_a, p_b = str(p_0), str(p_1)
                p_a_idx, p_b_idx = p_0_idx, p_1_idx
            else:
                p_a, p_b = str(p_1), str(p_0)
                p_a_idx, p_b_idx = p_1_idx, p_0_idx

            weight_mask_dict[p_a + p_b][p_a_idx, p_b_idx] = 1.

        # return weight_mask_dict
        nn_weight_mask_dict = nn.ParameterDict()
        for name, layer in weight_mask_dict.items():
            nn_weight_mask_dict[name] = nn.Parameter(layer.data, requires_grad=False)
        return nn_weight_mask_dict


if __name__ == "__main__":
    pRBM = PegasusRBM(1325)