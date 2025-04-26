"""
Pegasus PCD (4-partite block Gibbs Sampler)

Naive implementation using dictionary of weights, biases and partition states
TODO Tensorize the PRBM parameters and operations

Author : Abhi (abhishek@myumanitoba.ca)
"""
import torch
from models.rbm import pegasusRBM


class PGBS:
    """
    Pegasus GBS (4-partite block Gibbs) sampler

    Attributes:
        _prbm: an instance of CaloQVAE.models.rbm.pegasusRBM
        _batch_size: no. of block Gibbs chains
        _n_steps: no. of block Gibbs steps
    """

    def __init__(self, prbm: pegasusRBM.PegasusRBM, batch_size: int,
                 n_steps: int, **kwargs):
        """__init__()

        :param PRBM (object)    : Configuration of the Pegasus RBM
        :param batch_size (int) : No. of blocked Gibbs chains to run in
                                  parallel
        :param n_steps (int) : No. of MCMC steps
        :return None
        """
        super().__init__(**kwargs)
        self._batch_size = batch_size
        self._n_steps = n_steps
        self._prbm = prbm

    def _p_state(self, weights_ax, weights_bx, weights_cx,
                 pa_state, pb_state, pc_state, bias_x) -> torch.Tensor:
        """partition_state()

        :param weights_a (torch.Tensor) : (n_nodes_a, n_nodes_x)
        :param weights_b (torch.Tensor) : (n_nodes_b, n_nodes_x)
        :param weights_c (torch.Tensor) : (n_nodes_c, n_nodes_x)
        :param pa_state (torch.Tensor) : (batch_size, n_nodes_a)
        :param pb_state (torch.Tensor) : (batch_size, n_nodes_b)
        :param pc_state (torch.Tensor) : (batch_size, n_nodes_c)
        :param bias_x (torch.Tensor) : (n_nodes_x)
        """
        p_activations = (torch.matmul(pa_state, weights_ax) +
                         torch.matmul(pb_state, weights_bx) +
                         torch.matmul(pc_state, weights_cx) + bias_x)
        return torch.bernoulli(torch.sigmoid(p_activations))

    def block_gibbs_sampling(self, p0=None,p1=None,p2=None,p3=None, method='Rdm'):
        """block_gibbs_sampling()

        :return p0_state (torch.Tensor) : (batch_size, n_nodes_p1)
        :return p1_state (torch.Tensor) : (batch_size, n_nodes_p2)
        :return p2_state (torch.Tensor) : (batch_size, n_nodes_p3)
        :return p3_state (torch.Tensor) : (batch_size, n_nodes_p4)
        """
        prbm = self._prbm
        p0_bias = prbm._bias_dict['0']
        p_bias = prbm.bias_dict
        p_weight = prbm.weight_dict
        p0_bias = p_bias['0']
        p1_bias = p_bias['1']
        p2_bias = p_bias['2']
        p3_bias = p_bias['3']

        
        if method == 'Rdm' or p1==None:
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
        elif method == 'CD':
            # Initialize the random state of partitions 1, 2, and 3
            p1_state = p1
            p2_state = p2
            p3_state = p3
        elif method == 'PCD':
            pass
            
        for _ in range(self._n_steps):
            p0_state = self._p_state(p_weight['01'].T,
                                     p_weight['02'].T,
                                     p_weight['03'].T,
                                     p1_state, p2_state, p3_state,
                                     p0_bias)
            p1_state = self._p_state(p_weight['01'],
                                     p_weight['12'].T,
                                     p_weight['13'].T,
                                     p0_state, p2_state, p3_state,
                                     p1_bias)
            p2_state = self._p_state(p_weight['02'],
                                     p_weight['12'],
                                     p_weight['23'].T,
                                     p0_state, p1_state, p3_state,
                                     p2_bias)
            p3_state = self._p_state(p_weight['03'],
                                     p_weight['13'],
                                     p_weight['23'],
                                     p0_state, p1_state, p2_state,
                                     p3_bias)

        return p0_state.detach(), p1_state.detach(), \
            p2_state.detach(), p3_state.detach()
    
    def block_gibbs_sampling_cond(self, p0,p1=None,p2=None,p3=None, method='Rdm'):
        """block_gibbs_sampling()

        :return p0_state (torch.Tensor) : (batch_size, n_nodes_p1)
        :return p1_state (torch.Tensor) : (batch_size, n_nodes_p2)
        :return p2_state (torch.Tensor) : (batch_size, n_nodes_p3)
        :return p3_state (torch.Tensor) : (batch_size, n_nodes_p4)
        """
        prbm = self._prbm
        p0_bias = prbm._bias_dict['0']
        p_bias = prbm.bias_dict
        p_weight = prbm.weight_dict
        # p0_bias = p_bias['0']
        p1_bias = p_bias['1']
        p2_bias = p_bias['2']
        p3_bias = p_bias['3']

        
        if method == 'Rdm' or p1==None:
            # Initialize the random state of partitions 1, 2, and 3
            p1_state = torch.bernoulli(torch.rand(p0.shape[0], #self._batch_size,
                                                  prbm.nodes_per_partition,
                                                  device=p1_bias.device))
            p2_state = torch.bernoulli(torch.rand(p0.shape[0], #self._batch_size,
                                                  prbm.nodes_per_partition,
                                                  device=p1_bias.device))
            p3_state = torch.bernoulli(torch.rand(p0.shape[0], #self._batch_size,
                                              prbm.nodes_per_partition,
                                              device=p1_bias.device))
            # p3_state = p3.to(p1_state.dtype)
        elif method == 'CD':
            # Initialize the random state of partitions 1, 2, and 3
            p1_state = p1
            p2_state = p2
            p3_state = p3
        elif method == 'PCD':
            pass
            
        for _ in range(self._n_steps):
            # p0_state = self._p_state(p_weight['01'].T,
            #                          p_weight['02'].T,
            #                          p_weight['03'].T,
            #                          p1_state, p2_state, p3_state,
            #                          p0_bias)
            p1_state = self._p_state(p_weight['01'],
                                     p_weight['12'].T,
                                     p_weight['13'].T,
                                     p0, p2_state, p3_state,
                                     p1_bias)
            p2_state = self._p_state(p_weight['02'],
                                     p_weight['12'],
                                     p_weight['23'].T,
                                     p0, p1_state, p3_state,
                                     p2_bias)
            p3_state = self._p_state(p_weight['03'],
                                     p_weight['13'],
                                     p_weight['23'],
                                     p0, p1_state, p2_state,
                                     p3_bias)

        return p0.detach(), p1_state.detach(), \
            p2_state.detach(), p3_state.detach()

    @property
    def batch_size(self):
        """Accessor method for a protected variable

        :return batch_size of the BGS samples
        """
        return self._batch_size
    
    def gradient_rbm(self, post_samples, n_nodes_p, rbmMethod):
        #Gen data for gradient
        post_zetas = torch.cat(post_samples, 1)
        data_mean = post_zetas.mean(dim=0)
        torch.clamp_(data_mean, min=1e-4, max=(1. - 1e-4))
        vh_data_mean = (post_zetas.transpose(0,1) @ post_zetas) / post_zetas.size(0)

        p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_cond(post_zetas[:, :n_nodes_p],
                                             post_zetas[:, n_nodes_p:2*n_nodes_p],
                                             post_zetas[:, 2*n_nodes_p:3*n_nodes_p],
                                             post_zetas[:, 3*n_nodes_p:], method=rbmMethod)

        post_zetas_gen = torch.cat([p0_state,p1_state,p2_state,p3_state], dim=1)
        data_gen = post_zetas_gen.mean(dim=0)
        torch.clamp_(data_gen, min=1e-4, max=(1. - 1e-4));
        vh_gen_mean = (post_zetas_gen.transpose(0,1) @ post_zetas_gen) / post_zetas_gen.size(0)
        
        # compute gradient
        self.grad = {"bias": {}, "weight":{}}
        for i in range(4):
            self.grad["bias"][str(i)] = data_mean[n_nodes_p*i:n_nodes_p*(i+1)] - data_gen[n_nodes_p*i:n_nodes_p*(i+1)]
            
        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self.grad["weight"][str(i)+str(j)] = (vh_data_mean[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)] - vh_gen_mean[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)]) * self._prbm._weight_mask_dict[str(i)+str(j)]

        # for i in range(3):
        #     for j in [0,1,2,3]:
        #         if j > i:
        #             self.grad["weight"][str(i)+str(j)] = (data_mean[n_nodes_p*i:n_nodes_p*(i+1)].unsqueeze(1) @ data_mean[n_nodes_p*j:n_nodes_p*(j+1)].unsqueeze(0) - data_gen[n_nodes_p*i:n_nodes_p*(i+1)].unsqueeze(1) @ data_gen[n_nodes_p*j:n_nodes_p*(j+1)].unsqueeze(0)) * self._prbm._weight_mask_dict[str(i)+str(j)]

    def gradient_rbm_centered(self, post_samples, n_nodes_p, rbmMethod):
        #Gen data for gradient
        post_zetas = torch.cat(post_samples, 1)
        data_mean = post_zetas.mean(dim=0)
        torch.clamp_(data_mean, min=1e-4, max=(1. - 1e-4))
        vh_data_cov = torch.cov(post_zetas.T)

        p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_cond(post_zetas[:, :n_nodes_p],
                                             post_zetas[:, n_nodes_p:2*n_nodes_p],
                                             post_zetas[:, 2*n_nodes_p:3*n_nodes_p],
                                             post_zetas[:, 3*n_nodes_p:], method=rbmMethod)

        post_zetas_gen = torch.cat([p0_state,p1_state,p2_state,p3_state], dim=1)
        data_gen = post_zetas_gen.mean(dim=0)
        torch.clamp_(data_gen, min=1e-4, max=(1. - 1e-4));
        vh_gen_cov = torch.cov(post_zetas_gen.T)

        # compute gradient
        self.grad = {"bias": {}, "weight":{}}
        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self.grad["weight"][str(i)+str(j)] = (vh_data_cov[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)] - vh_gen_cov[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)]) * self._prbm._weight_mask_dict[str(i)+str(j)]
                    
        
        for i in range(4):
            self.grad["bias"][str(i)] = data_mean[n_nodes_p*i:n_nodes_p*(i+1)] - data_gen[n_nodes_p*i:n_nodes_p*(i+1)]
            for j in range(4):
                if j > i:
                    self.grad["bias"][str(i)] = self.grad["bias"][str(i)] - 0.5 * torch.matmul(self.grad["weight"][str(i)+str(j)], (data_mean[n_nodes_p*j:n_nodes_p*(j+1)] + data_gen[n_nodes_p*j:n_nodes_p*(j+1)]))
                elif j < i:
                    self.grad["bias"][str(i)] = self.grad["bias"][str(i)] - 0.5 * torch.matmul(self.grad["weight"][str(j)+str(i)].T , (data_mean[n_nodes_p*j:n_nodes_p*(j+1)] + data_gen[n_nodes_p*j:n_nodes_p*(j+1)]))
    
    def update_params(self, lr=0.01):
        for i in range(4):
            self._prbm.bias_dict[str(i)] = self._prbm.bias_dict[str(i)] + lr * self.grad["bias"][str(i)]

        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self._prbm.weight_dict[str(i)+str(j)] = self._prbm.weight_dict[str(i)+str(j)] + lr * self.grad["weight"][str(i)+str(j)]