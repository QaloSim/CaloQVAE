import torch
import numpy as np

class Stats():
    """
    This module is for  CRBM stats generation
    """
    def __init__(self, sampler, batch_size=None, n_steps=None):
        self._rbm = sampler.rbm
        if batch_size==None:
            self._batch_size = sampler._batch_size
        else:
            self._batch_size = batch_size
        if n_steps==None:
            self._n_steps = sampler.n_gibbs_sampling_steps
        else:
            self._n_steps = n_steps
        
    def _p_state_ais(self, weights, p_state, bias, beta) -> torch.Tensor:
        """partition_state()

        :param weights (torch.Tensor) : (n_nodes_a, n_nodes_x)
        :param p_state (torch.Tensor) : (batch_size, n_nodes_a)
        :param bias (torch.Tensor) : (n_nodes_x)
        """
        p_activations = (torch.matmul(p_state, weights * beta) + bias)
        return torch.bernoulli(torch.sigmoid(p_activations)).detach()
    
    
    def block_gibbs_sampling_ais(self, beta, v_state=None, h_state=None):
        """block_gibbs_sampling()

        :return v_state (torch.Tensor) : (batch_size, n_nodes_v)
        :return h_state (torch.Tensor) : (batch_size, n_nodes_h)
        """
        rbm = self._rbm
        
        weights = rbm.weights
        v_bias = rbm.visible_bias
        h_bias = rbm.hidden_bias

        if v_state == None:
            # Initialize the random state of hidden variables
            h_state = torch.bernoulli(torch.rand(self._batch_size,
                                                  rbm.hidden_bias.size()[0],
                                                  device=v_bias.device))
        else:
            v_state = v_state.to(v_bias.device)
            h_state = h_state.to(v_bias.device)
            
        for _ in range(self._n_steps):
            v_state = self._p_state_ais(weights.T,
                                     h_state,
                                     v_bias, beta).detach()
            h_state = self._p_state_ais(weights,
                                     v_state,
                                     h_bias, beta).detach()

        return v_state.detach(), h_state.detach()
    
    def energy_samples(self, v_state, h_state, beta):
        """Energy expectation value under the 4-partite BM
        Overrides energy_exp in gumbolt.py

        :param v_state (torch.Tensor) : (batch_size, n_nodes_v)
        :param h_state (torch.Tensor) : (batch_size, n_nodes_h)

        :return energy expectation value over the current batch
        """
        weights = self._rbm.weights
        v_bias = self._rbm.visible_bias
        h_bias = self._rbm.hidden_bias

        w_dict_cp = {}

        # Broadcast weight matrix (n_nodes_v, n_nodes_h) to
        # (batch_size, n_nodes_v, n_nodes_h)
        weights_br = weights + torch.zeros((v_state.size(0),) +
                                            weights.size(),
                                            device=weights.device)
        

        # Prepare v_state_t for torch.bmm()
        # Change v_state.size() to (batch_size, 1, n_nodes_v)
        v_state_t = v_state.unsqueeze(2).permute(0, 2, 1)
        #p1_state_t = p1_state.unsqueeze(2).permute(0, 2, 1)
        #p2_state_t = p2_state.unsqueeze(2).permute(0, 2, 1)

        # Prepare h_state for torch.bmm()
        # Change h_state.size() to (batch_size, n_nodes_h, 1)
        h_state_i = h_state.unsqueeze(2)
        #p2_state_i = p2_state.unsqueeze(2)
        #p3_state_i = p3_state.unsqueeze(2)

        # Compute the energies for batch samples
        batch_energy = -torch.matmul(v_state, v_bias) - \
            torch.matmul(h_state, h_bias) - \
            torch.bmm(v_state_t,
                      torch.bmm(beta * weights_br, h_state_i)).reshape(-1)

        return batch_energy.detach()
    
    
    def AIS(self, nbeta=20.0):
        # http://www.cs.utoronto.ca/~rsalakhu/papers/bm.pdf
        self.lnZa = np.sum([torch.log(1 + torch.exp(-self._rbm.visible_bias)).sum().item(), torch.log(1 + torch.exp(-self._rbm.hidden_bias)).sum().item()])
        FreeEnergy_ratios = 0.0
        Δbeta = 1/nbeta
        for beta in np.arange(0.0,1.0,Δbeta):
            if beta == 0:
                v_state, h_state = self.block_gibbs_sampling_ais(beta)
            else:
                v_state, h_state = self.block_gibbs_sampling_ais(beta, v_state, h_state)
            energy_samples_i = self.energy_samples(v_state, h_state, beta)
            energy_samples_i_plus = self.energy_samples(v_state, h_state, beta+Δbeta)
            FreeEnergy_ratios = FreeEnergy_ratios + torch.log(torch.exp(energy_samples_i - energy_samples_i_plus).mean())
        logZb = FreeEnergy_ratios + self.lnZa
        return logZb
    
    def RAIS(self, nbeta=20.0):
        self.lnZb = np.sum([torch.log(1 + torch.exp(-self._rbm.visible_bias)).sum().item(), torch.log(1 + torch.exp(-self._rbm.hidden_bias)).sum().item()])
        FreeEnergy_ratios = 0.0
        Δbeta = 1/nbeta

        # Reverse AIS: Start from the target distribution (beta = 1)
        for beta in np.arange(1.0, 0.0, -Δbeta):
            if beta == 1:
                v_state, h_state = self.block_gibbs_sampling_ais(beta)
            else:
                # When beta is not 1, continue the sampling from the current state
                v_state, h_state = self.block_gibbs_sampling_ais(beta, v_state, h_state)

            # Calculate energies for the current beta and the next beta (which is beta - Δbeta)
            energy_samples_i = self.energy_samples(v_state, h_state, beta)
            energy_samples_i_minus = self.energy_samples(v_state, h_state, beta - Δbeta)

            # Accumulate the free energy differences
            FreeEnergy_ratios += torch.log(torch.exp(- energy_samples_i_minus + energy_samples_i).mean())

        # The final estimate for logZa (partition function of the base distribution)
        logZa = - FreeEnergy_ratios + self.lnZb
        return logZa