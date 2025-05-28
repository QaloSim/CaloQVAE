"""
PyTorch implementation of a quadripartite Boltzmann machine with a 
Zephyr/Advantage2 QPU topology
"""
import dwave_networkx as dnx
import networkx as nx
from dwave.system import DWaveSampler
from hybrid.decomposers import _chimeralike_to_zephyr

import itertools
import math

import torch
from torch import nn

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class ZephyrRBM(nn.Module):
    """
    PyTorch implementation of a quadripartite Boltzmann machine with a 
    Zephyr/Advantage2 QPU topology
    """
    def __init__(self, nodes_per_partition: int, fullyconnected: bool = False, **kwargs):
        """__init__()

        Initialize an instance of a 4-partite PegasusRBM

        :param nodes_per_layer (int) : Number of nodes for each partition
        :param qpu (bool) : Only allow connections present on the QPU 
        """
        super(ZephyrRBM, self).__init__(**kwargs)

        # RBM constants
        self._n_partitions = 4

        # Emtpy RBM parameters
        self._weight_dict = nn.ParameterDict()
        self._bias_dict = nn.ParameterDict()
        self._nodes_per_partition = nodes_per_partition
        # if self._nodes_per_partition > 302 and fullyconnected == False:
        #     logger.warn("No more than 302 nodes per partition for Adv2 \
        #                 at the time being. Will set partitions to 302. \
        #                 Otherwise, stop job, reduce nodes per partition and restart.")
        #     # fullyconnected = True
        #     self._nodes_per_partition = 302
        self._weight_mask_dict = nn.ParameterDict()

        # Fully-connected or Pegasus-restricted 4-partite BM
        # self._qpu = qpu
        self.fullyconnected = fullyconnected

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

        # if qpu:
        self._qubit_idx_dict, device = self.gen_qubit_idx_dict()
        if not fullyconnected:
            self._weight_mask_dict = self.gen_weight_mask_dict(
                self._qubit_idx_dict, device)
        else:
            for key in itertools.combinations(range(self._n_partitions), 2):
                str_key = ''.join([str(key[i]) for i in range(len(key))])
                self._weight_mask_dict[str_key] = nn.Parameter(torch.ones(self._nodes_per_partition, self._nodes_per_partition), requires_grad=False)

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
        # if self._qpu:
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
        self.load_coordinates()

        # Initialize the dict to be returned
        idx_dict = {}
        for partition in range(self._n_partitions):
            idx_dict[str(partition)] = []

        for q in self.coordinated_graph.nodes:
            _idx = (2*q[0]+q[1] + 2*q[4]+q[3])%4
            idx_dict[str(_idx)].append(self.coordinates_to_idx(q, self.m,self.t))
            
        for partition, idxs in idx_dict.items():
            idx_dict[partition] = idx_dict[partition][:
                self._nodes_per_partition]
            
        self.idx_dict = idx_dict
        return idx_dict, self._qpu_sampler

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
    
        """
            A simpler way to build the mask:
            for i, partition_a in enumerate(self.idx_dict.keys()):
                for qubit_a in self.idx_dict[partition_a]:
                    for qubit_b in self._qpu_sampler.adjacency[qubit_a]:
                        for partition_b in list(self.idx_dict.keys())[i:]:
                            if qubit_b in self.idx_dict[partition_b]:
                                weight_idx = partition_a + partition_b
                                idx_a = self.idx_dict[partition_a].index(qubit_a)
                                idx_b = self.idx_dict[partition_b].index(qubit_b)
                                self._weight_mask_dict[weight_idx][idx_a,idx_b] = 1.0
        """

    def load_coordinates(self):
        try:
            self._qpu_sampler = DWaveSampler(solver={'topology__type': 'zephyr', 'chip_id':'Advantage2_system1.1'})
        except:
            self._qpu_sampler = DWaveSampler(solver={'topology__type': 'zephyr', 'chip_id':'Advantage2_system2.6'})
            logger.warn("Switching to Zephyr prototype. No more than 302 nodes per partition for Adv2 \
                     at the time being. Will set partitions to 302. \
                     Otherwise, stop job, reduce nodes per partition and restart.")
            self._nodes_per_partition=302
        self.m, self.t = self._qpu_sampler.properties['topology']['shape']
        graph = dnx.zephyr_graph(m=self.m, t=self.t,
                             node_list=self._qpu_sampler.nodelist, edge_list=self._qpu_sampler.edgelist)
        self.coordinated_graph = nx.relabel_nodes(
                graph,
                {q: dnx.zephyr_coordinates(self.m,self.t).linear_to_zephyr(q)
                 for q in graph.nodes})
        
    def coordinates_to_idx(self, q, m, t):
        return q[4] + m*(q[3] + 2*(q[2] + t*(q[1]+(2*m+1)*q[0])))

if __name__ == "__main__":
    pRBM = PegasusRBM(1325)