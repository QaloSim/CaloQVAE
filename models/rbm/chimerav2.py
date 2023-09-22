"""
Chimera adapted to Pegasus architecture.
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

class Chim(RBM):
    def __init__(self, n_visible, n_hidden, bernoulli=False, **kwargs):
        super(Chim, self).__init__(n_visible, n_hidden, **kwargs)

        self._n_visible=n_visible
        self._n_hidden=n_hidden
        
        require_grad=True
        
        n_cells = max(math.ceil(n_visible/_CELL_SIDE_QUBITS), math.ceil(n_hidden/_CELL_SIDE_QUBITS))
        n_rows = math.ceil(math.sqrt(n_cells))
        n_cols = n_rows
        
        assert n_cols<=_MAX_ROW_COLS
        
        visible_qubit_idxs = []
        hidden_qubit_idxs = []
        
#         qpu_sampler = DWaveSampler(solver={"topology__type":"chimera", "chip_id":"DW_2000Q_6"})
        qpu_sampler = DWaveSampler(solver={"topology__type":"pegasus"})
        qpu_nodes = qpu_sampler.nodelist
        qpu_edges = qpu_sampler.edgelist
        
        self.qpu_nodes = qpu_nodes
        self.qpu_edges = qpu_edges
        
        self.buildAdjMatrix(14)
        self.build4pGraph()
        self.checkNoOverlap()
        self.buildRMBToQPUMapping()
        self.buildWeightMasks()
        
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
        
    def buildAdjMatrix(self, con_thrsh=15):
        # ar is the adj matrix
        ar = torch.zeros(np.max(self.qpu_nodes)+1, np.max(self.qpu_nodes)+1)
        # It's not clear if adjacency implies connection. Better to rely on edgelist = active couplings
        # for node in self.qpu_nodes:
        #     for adj in qpu_sampler.adjacency[node]:
        #         ar[node, adj] = 1
        #         ar[adj, node] = 1

        for (x,y) in self.qpu_edges:
            ar[x, y] = 1
            ar[y, x] = 1

        # We select qbits with 15 connections each. ~4k
        a = ar.sum(dim=1).sort()
        self.arIdx15 = list(a.indices[a.values >= con_thrsh].numpy())
        
    def build4pGraph(self):
        # Each key will hold the idx of qubits belonging to each partition
        # The idea is to first find all qubits of partition 1 and put everything else in remainder
        # Then repeat the process for partition 2 and 3.
        FourpGraph = dict()
        FourpGraph["1"] = []
        FourpGraph["2"] = []
        FourpGraph["3"] = []
        FourpGraph["4"] = []

        remainder0 = self.arIdx15 
        l = []

        ps = ["1","2","3","4"]
        for j,p in enumerate(ps):
            for i in remainder0:
                if FourpGraph[p] == []: # First element in remainder0 is stored in partition. The adjacent is stored in remainder
                    FourpGraph[p].append(i)
                else:
                    # check if there's an edge between i and the elems in partition being built
                    checkTuples = [item for item in self.qpu_edges if i in item]
                    for j in FourpGraph[p]:
                        l = l + [item for item in checkTuples if j in item]
                    if l==[]:
                        FourpGraph[p].append(i)
                l = []
            remainder0 = list(np.setdiff1d(np.array(remainder0), np.array(FourpGraph[p])))
            assert len(list(np.unique(FourpGraph[p]))) == len(FourpGraph[p])
            FourpGraph[p].sort
        self.FourpGraph = FourpGraph
        self.ps = ps
        
    def checkNoOverlap(self):
        # Checks that no qbit in a given partition has connections with qbits in same partition
        # If nothing gets printed => Good!
        # for p in self.ps:
        #     for i in self.FourpGraph[p]:
        #         for j in self.FourpGraph[p]:
        #             if i != j:
        #                 checkTuples = [item for item in self.qpu_edges if i in item]
        #                 checkTuples2 = [item for item in checkTuples if j in item]
        #                 assert checkTuples2 == []

        for p in self.ps:
            for q in self.ps:
                if p != q:
                    assert len(list(np.setdiff1d(np.array(self.FourpGraph[p]), np.array(self.FourpGraph[q])))) == len(self.FourpGraph[p])
        
        
    def buildRMBToQPUMapping(self):
        self.RBMtoQPUIdx = dict()
        for p in self.ps:
            tmp = dict()
            for i, qbitIdx in enumerate(self.FourpGraph[p]):
                tmp[qbitIdx] = i
            self.RBMtoQPUIdx[p] = tmp
            
    def buildWeightMasks(self):
        # We build RBM masking
        self.weight_mask = dict()
        self.weight_mask["01"] = torch.zeros(len(self.FourpGraph["1"]), len(self.FourpGraph["2"]))
        self.weight_mask["02"] = torch.zeros(len(self.FourpGraph["1"]), len(self.FourpGraph["3"]))
        self.weight_mask["03"] = torch.zeros(len(self.FourpGraph["1"]), len(self.FourpGraph["4"]))

        self.weight_mask["12"] = torch.zeros(len(self.FourpGraph["2"]), len(self.FourpGraph["3"]))
        self.weight_mask["13"] = torch.zeros(len(self.FourpGraph["2"]), len(self.FourpGraph["4"]))

        self.weight_mask["23"] = torch.zeros(len(self.FourpGraph["3"]), len(self.FourpGraph["4"]))


        c = []
        for (x,y) in self.qpu_edges:
            if x in self.arIdx15 and y in self.arIdx15:
                if x in self.FourpGraph["1"]:
                    n = 0
                    pIdx = "1"
                elif x in self.FourpGraph["2"]:
                    n = 1
                    pIdx = "2"
                elif x in self.FourpGraph["3"]:
                    n = 2
                    pIdx = "3"
                elif x in self.FourpGraph["4"]:
                    n = 3
                    pIdx = "4"
                else:
                    c.append(x)
                    n = -1
                    pIdx = "0"

                if y in self.FourpGraph["1"]:
                    m = 0
                    qIdx = "1"
                elif y in self.FourpGraph["2"]:
                    m = 1
                    qIdx = "2"
                elif y in self.FourpGraph["3"]:
                    m = 2
                    qIdx = "3"
                elif y in self.FourpGraph["4"]:
                    m = 3
                    qIdx = "4"
                else:
                    c.append(y)
                    m = -1
                    qIdx = "0"

                if n != -1 and m != -1 and m != n and qIdx != "0" and pIdx != "0":
                    key = str(min(n,m)) + str(max(n,m))
                    if int(pIdx) < int(qIdx):
                        self.weight_mask[key][self.RBMtoQPUIdx[pIdx][x], self.RBMtoQPUIdx[qIdx][y]] = 1
                    else:
                        self.weight_mask[key][self.RBMtoQPUIdx[qIdx][y], self.RBMtoQPUIdx[pIdx][x]] = 1

        # c = list(np.unique(c))
        # assert remainder0 == c
        

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

