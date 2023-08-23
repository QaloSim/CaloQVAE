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

    def block_gibbs_sampling(self):
        """block_gibbs_sampling()

        :return p0_state (torch.Tensor) : (batch_size, n_nodes_p1)
        :return p1_state (torch.Tensor) : (batch_size, n_nodes_p2)
        :return p2_state (torch.Tensor) : (batch_size, n_nodes_p3)
        :return p3_state (torch.Tensor) : (batch_size, n_nodes_p4)
        """
        prbm = self._prbm
        p0_bias = prbm._bias_dict['0']

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

#         for _ in range(self._n_steps):
#             p0_state = self._p_state(prbm._weight_dict['01'].T,
#                                      prbm._weight_dict['02'].T,
#                                      prbm._weight_dict['03'].T,
#                                      p1_state, p2_state, p3_state,
#                                      p0_bias)
#             p1_state = self._p_state(prbm._weight_dict['01'],
#                                      prbm._weight_dict['02'].T,
#                                      prbm._weight_dict['03'].T,
#                                      p0_state, p2_state, p3_state,
#                                      prbm._bias_dict['1'])
#             p2_state = self._p_state(prbm._weight_dict['01'],
#                                      prbm._weight_dict['02'],
#                                      prbm._weight_dict['03'].T,
#                                      p0_state, p1_state, p3_state,
#                                      prbm._bias_dict['2'])
#             p3_state = self._p_state(prbm._weight_dict['01'],
#                                      prbm._weight_dict['02'],
#                                      prbm._weight_dict['03'],
#                                      p0_state, p1_state, p2_state,
#                                      prbm._bias_dict['3'])
            
        for _ in range(self._n_steps):
            p0_state = self._p_state(prbm._weight_dict['01'].T,
                                     prbm._weight_dict['02'].T,
                                     prbm._weight_dict['03'].T,
                                     p1_state, p2_state, p3_state,
                                     p0_bias)
            p1_state = self._p_state(prbm._weight_dict['01'],
                                     prbm._weight_dict['12'].T,
                                     prbm._weight_dict['13'].T,
                                     p0_state, p2_state, p3_state,
                                     prbm._bias_dict['1'])
            p2_state = self._p_state(prbm._weight_dict['02'],
                                     prbm._weight_dict['12'],
                                     prbm._weight_dict['23'].T,
                                     p0_state, p1_state, p3_state,
                                     prbm._bias_dict['2'])
            p3_state = self._p_state(prbm._weight_dict['03'],
                                     prbm._weight_dict['13'],
                                     prbm._weight_dict['23'],
                                     p0_state, p1_state, p2_state,
                                     prbm._bias_dict['3'])

        return p0_state.detach(), p1_state.detach(), \
            p2_state.detach(), p3_state.detach()

    @property
    def batch_size(self):
        """Accessor method for a protected variable

        :return batch_size of the BGS samples
        """
        return self._batch_size
