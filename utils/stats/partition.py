import torch
import numpy as np

class Stats():
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
        
    def _p_state_ais(self, weights_ax, weights_bx, weights_cx, pa_state, pb_state, pc_state, bias_x, beta) -> torch.Tensor:
        """partition_state()

        :param weights_a (torch.Tensor) : (n_nodes_a, n_nodes_x)
        :param weights_b (torch.Tensor) : (n_nodes_b, n_nodes_x)
        :param weights_c (torch.Tensor) : (n_nodes_c, n_nodes_x)
        :param pa_state (torch.Tensor) : (batch_size, n_nodes_a)
        :param pb_state (torch.Tensor) : (batch_size, n_nodes_b)
        :param pc_state (torch.Tensor) : (batch_size, n_nodes_c)
        :param bias_x (torch.Tensor) : (n_nodes_x)
        """
        p_activations = (torch.matmul(pa_state, weights_ax * beta) +
                         torch.matmul(pb_state, weights_bx * beta) +
                         torch.matmul(pc_state, weights_cx * beta) + bias_x)
        return torch.bernoulli(torch.sigmoid(p_activations)).detach()
    
    
    def block_gibbs_sampling_ais(self, beta, p0_state=None, p1_state=None, p2_state=None, p3_state=None):
        """block_gibbs_sampling()

        :return p0_state (torch.Tensor) : (batch_size, n_nodes_p1)
        :return p1_state (torch.Tensor) : (batch_size, n_nodes_p2)
        :return p2_state (torch.Tensor) : (batch_size, n_nodes_p3)
        :return p3_state (torch.Tensor) : (batch_size, n_nodes_p4)
        """
        prbm = self._prbm
        p_bias = prbm.bias_dict
        p_weight = prbm.weight_dict
        p0_bias = p_bias['0']
        p1_bias = p_bias['1']
        p2_bias = p_bias['2']
        p3_bias = p_bias['3']

        if p0_state == None:
            # Initialize the random state of partitions 1, 2, and 3
            p1_state = torch.bernoulli(torch.rand(self._batch_size,
                                                  prbm.nodes_per_partition,
                                                  device=p0_bias.device))
            p2_state = torch.bernoulli(torch.rand(self._batch_size,
                                                  prbm.nodes_per_partition,
                                                  device=p0_bias.device))
            p3_state = torch.bernoulli(torch.rand(self._batch_size,
                                                  prbm.nodes_per_partition,
                                                  device=p0_bias.device))
        else:
            p0_state = p0_state.to(p0_bias.device)
            p1_state = p1_state.to(p0_bias.device)
            p2_state = p2_state.to(p0_bias.device)
            p3_state = p3_state.to(p0_bias.device)
            
        for _ in range(self._n_steps):
            p0_state = self._p_state_ais(p_weight['01'].T,
                                     p_weight['02'].T,
                                     p_weight['03'].T,
                                     p1_state, p2_state, p3_state,
                                     p0_bias, beta).detach()
            p1_state = self._p_state_ais(p_weight['01'],
                                     p_weight['12'].T,
                                     p_weight['13'].T,
                                     p0_state, p2_state, p3_state,
                                     p1_bias, beta).detach()
            p2_state = self._p_state_ais(p_weight['02'],
                                     p_weight['12'],
                                     p_weight['23'].T,
                                     p0_state, p1_state, p3_state,
                                     p2_bias, beta).detach()
            p3_state = self._p_state_ais(p_weight['03'],
                                     p_weight['13'],
                                     p_weight['23'],
                                     p0_state, p1_state, p2_state,
                                     p3_bias, beta).detach()

        return p0_state.detach(), p1_state.detach(), p2_state.detach(), p3_state.detach()
    
    def energy_samples(self, p0_state, p1_state, p2_state, p3_state, beta):
        """Energy expectation value under the 4-partite BM
        Overrides energy_exp in gumbolt.py

        :param p0_state (torch.Tensor) : (batch_size, n_nodes_p1)
        :param p1_state (torch.Tensor) : (batch_size, n_nodes_p2)
        :param p2_state (torch.Tensor) : (batch_size, n_nodes_p3)
        :param p3_state (torch.Tensor) : (batch_size, n_nodes_p4)

        :return energy expectation value over the current batch
        """
        # w_dict = self.prior.weight_dict
        # b_dict = self.prior.bias_dict
        w_dict = self._prbm.weight_dict
        b_dict = self._prbm.bias_dict

        w_dict_cp = {}

        # Broadcast weight matrices (n_nodes_pa, n_nodes_pb) to
        # (batch_size, n_nodes_pa, n_nodes_pb)
        for key in w_dict.keys():
            w_dict_cp[key] = w_dict[key] + torch.zeros((p0_state.size(0),) +
                                                    w_dict[key].size(),
                                                    device=w_dict[key].device)

        # Prepare px_state_t for torch.bmm()
        # Change px_state.size() to (batch_size, 1, n_nodes_px)
        p0_state_t = p0_state.unsqueeze(2).permute(0, 2, 1)
        p1_state_t = p1_state.unsqueeze(2).permute(0, 2, 1)
        p2_state_t = p2_state.unsqueeze(2).permute(0, 2, 1)

        # Prepare py_state for torch.bmm()
        # Change py_state.size() to (batch_size, n_nodes_py, 1)
        p1_state_i = p1_state.unsqueeze(2)
        p2_state_i = p2_state.unsqueeze(2)
        p3_state_i = p3_state.unsqueeze(2)

        # Compute the energies for batch samples
        batch_energy = -torch.matmul(p0_state, b_dict['0']) - \
            torch.matmul(p1_state, b_dict['1']) - \
            torch.matmul(p2_state, b_dict['2']) - \
            torch.matmul(p3_state, b_dict['3']) - \
            torch.bmm(p0_state_t,
                      torch.bmm(beta * w_dict_cp['01'], p1_state_i)).reshape(-1) - \
            torch.bmm(p0_state_t,
                      torch.bmm(beta * w_dict_cp['02'], p2_state_i)).reshape(-1) - \
            torch.bmm(p0_state_t,
                      torch.bmm(beta * w_dict_cp['03'], p3_state_i)).reshape(-1) - \
            torch.bmm(p1_state_t,
                      torch.bmm(beta * w_dict_cp['12'], p2_state_i)).reshape(-1) - \
            torch.bmm(p1_state_t,
                      torch.bmm(beta * w_dict_cp['13'], p3_state_i)).reshape(-1) - \
            torch.bmm(p2_state_t,
                      torch.bmm(beta * w_dict_cp['23'], p3_state_i)).reshape(-1)

        return batch_energy.detach()
    
    
    def AIS(self, nbeta=20.0):
        # http://www.cs.utoronto.ca/~rsalakhu/papers/bm.pdf
        self.lnZa = np.sum([torch.log(1 + torch.exp(-self._prbm.bias_dict[i] )).sum().item() for i in ['0','1','2','3']])
        FreeEnergy_ratios = 0.0
        Δbeta = 1/nbeta
        for beta in np.arange(0.0,1.0,Δbeta):
            if beta == 0:
                p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_ais(beta)
            else:
                p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_ais(beta, p0_state, p1_state, p2_state, p3_state)
            energy_samples_i = self.energy_samples(p0_state, p1_state, p2_state, p3_state, beta)
            energy_samples_i_plus = self.energy_samples(p0_state, p1_state, p2_state, p3_state, beta+Δbeta)
            FreeEnergy_ratios = FreeEnergy_ratios + torch.log(torch.exp(energy_samples_i - energy_samples_i_plus).mean())
        logZb = FreeEnergy_ratios + self.lnZa
        return logZb
    
    def RAIS(self, nbeta=20.0):
        self.lnZb = np.sum([torch.log(1 + torch.exp(-self._prbm.bias_dict[i])).sum().item() for i in ['0','1','2','3']])
        FreeEnergy_ratios = 0.0
        Δbeta = 1/nbeta

        # Reverse AIS: Start from the target distribution (beta = 1)
        for beta in np.arange(1.0, 0.0, -Δbeta):
            if beta == 1:
                p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_ais(beta)
            else:
                # When beta is not 1, continue the sampling from the current state
                p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_ais(beta, p0_state, p1_state, p2_state, p3_state)

            # Calculate energies for the current beta and the next beta (which is beta - Δbeta)
            energy_samples_i = self.energy_samples(p0_state, p1_state, p2_state, p3_state, beta)
            energy_samples_i_minus = self.energy_samples(p0_state, p1_state, p2_state, p3_state, beta - Δbeta)

            # Accumulate the free energy differences
            FreeEnergy_ratios += torch.log(torch.exp(- energy_samples_i_minus + energy_samples_i).mean())

        # The final estimate for logZa (partition function of the base distribution)
        logZa = - FreeEnergy_ratios + self.lnZb
        return logZa
