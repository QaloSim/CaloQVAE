import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from CaloQVAE import logging
logger = logging.getLogger(__name__)

class CRBMStats():
    """
    This module is for  PRBM stats generation
    """
    def __init__(self, sampler, batch_size=None, n_steps=None):
        self._prbm = sampler._prbm
        if batch_size==None:
            self._batch_size = sampler._batch_size
        else:
            self._batch_size = batch_size
        if n_steps==None:
            self._n_steps = sampler._n_steps
        else:
            self._n_steps = n_steps
    
    
    
    def energy_samples(self, u, p1_state, p2_state, p3_state, beta):
        """Energy Samples under the 4-partite CRBM

        :return u.       (torch.Tensor) : (batch_size, n_nodes_u)
        :return p1_state (torch.Tensor) : (batch_size, n_nodes_p1)
        :return p2_state (torch.Tensor) : (batch_size, n_nodes_p2)
        :return p3_state (torch.Tensor) : (batch_size, n_nodes_p3)

        :return energy samples
        """
        w_dict = self._prbm.weight_dict
        b_dict = self._prbm.bias_dict

        w_dict_cp = {}

        # Broadcast weight matrices (n_nodes_pa, n_nodes_pb) to
        # (batch_size, n_nodes_pa, n_nodes_pb)
        for key in w_dict.keys():
            w_dict_cp[key] = w_dict[key] + torch.zeros((p1_state.size(0),) +
                                                    w_dict[key].size(),
                                                    device=w_dict[key].device)

        # Prepare px_state_t for torch.bmm()
        # Change px_state.size() to (batch_size, 1, n_nodes_px)
        u_t = u.unsqueeze(2).permute(0, 2, 1)
        p1_state_t = p1_state.unsqueeze(2).permute(0, 2, 1)
        p2_state_t = p2_state.unsqueeze(2).permute(0, 2, 1)

        # Prepare py_state for torch.bmm()
        # Change py_state.size() to (batch_size, n_nodes_py, 1)
        p1_state_i = p1_state.unsqueeze(2)
        p2_state_i = p2_state.unsqueeze(2)
        p3_state_i = p3_state.unsqueeze(2)
        
        dev = b_dict['1'].device
        
        u_t = u_t.to(dev)
        p1_state_t.to(dev)
        p2_state_t.to(dev)
        p1_state_i.to(dev)
        p2_state_i.to(dev)
        p3_state_i.to(dev)
        '''
        print(b_dict['1'].device)
        print(b_dict['2'].device)
        print(b_dict['3'].device)
        print(w_dict_cp['01'].device)
        print(w_dict_cp['02'].device)
        print(w_dict_cp['03'].device)
        print(w_dict_cp['12'].device)
        print(w_dict_cp['13'].device)
        print(w_dict_cp['23'].device)
        print("#####")
        print(u_t.device)
        print(p1_state_t.device)
        print(p2_state_t.device)
        print(p1_state_i.device)
        print(p2_state_i.device)
        print(p3_state_i.device)
        '''

        # Compute the energies for batch samples
        batch_energy = -torch.matmul(p1_state, b_dict['1']) - \
            torch.matmul(p2_state, b_dict['2']) - \
            torch.matmul(p3_state, b_dict['3']) - \
            torch.bmm(u_t,
                      torch.bmm(beta * w_dict_cp['01'], p1_state_i)).reshape(-1) - \
            torch.bmm(u_t,
                      torch.bmm(beta * w_dict_cp['02'], p2_state_i)).reshape(-1) - \
            torch.bmm(u_t,
                      torch.bmm(beta * w_dict_cp['03'], p3_state_i)).reshape(-1) - \
            torch.bmm(p1_state_t,
                      torch.bmm(beta * w_dict_cp['12'], p2_state_i)).reshape(-1) - \
            torch.bmm(p1_state_t,
                      torch.bmm(beta * w_dict_cp['13'], p3_state_i)).reshape(-1) - \
            torch.bmm(p2_state_t,
                      torch.bmm(beta * w_dict_cp['23'], p3_state_i)).reshape(-1)

        return batch_energy.detach()
    
