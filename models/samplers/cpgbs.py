"""
Conditioned Pegasus PCD (4-partite block Gibbs Sampler)

Naive implementation using dictionary of weights, biases and partition states
TODO Tensorize the PRBM parameters and operations

Author : Sebastian (sgonzalez@triumf.ca)
"""
import torch
from models.rbm import pegasusRBM
from models.samplers import pgbs


class CPGBS(pgbs.PGBS):
    """
    Conditioned Pegasus GBS (4-partite block Gibbs) sampler

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
        super(CPGBS).__init__(**kwargs)
        
        self._batch_size = batch_size
        self._n_steps = n_steps
        self._prbm = prbm

    def block_gibbs_sampling(self, u, p1=None,p2=None,p3=None, method='Rdm'):
        """block_gibbs_sampling()

        :return u.       (torch.Tensor) : (batch_size, n_nodes_u)
        :return p1_state (torch.Tensor) : (batch_size, n_nodes_p1)
        :return p2_state (torch.Tensor) : (batch_size, n_nodes_p2)
        :return p3_state (torch.Tensor) : (batch_size, n_nodes_p3)
        """
        prbm = self._prbm
        p_bias = prbm.bias_dict
        p_weight = prbm.weight_dict
        _ = p_bias['0'] # Ignore biases for p0 = u
        p1_bias = p_bias['1']
        p2_bias = p_bias['2']
        p3_bias = p_bias['3']
        
        u = u.to(p1_bias.device)

        
        if method == 'Rdm' or p1==None:
            # Initialize the random state of partitions 1, 2, and 3
            p1_state = torch.bernoulli(torch.rand(u.shape[0], #self._batch_size,
                                                  prbm.nodes_per_partition,
                                                  device=p1_bias.device))
            p2_state = torch.bernoulli(torch.rand(u.shape[0], #self._batch_size,
                                                  prbm.nodes_per_partition,
                                                  device=p1_bias.device))
            p3_state = torch.bernoulli(torch.rand(u.shape[0], #self._batch_size,
                                              prbm.nodes_per_partition,
                                              device=p1_bias.device))
        elif method == 'CD':
            # Initialize the random state of partitions 1, 2, and 3
            p1_state = p1
            p2_state = p2
            p3_state = p3
            
        for _ in range(self._n_steps):
            p1_state = self._p_state(p_weight['01'],
                                     p_weight['12'].T,
                                     p_weight['13'].T,
                                     u, p2_state, p3_state,
                                     p1_bias)
            p2_state = self._p_state(p_weight['02'],
                                     p_weight['12'],
                                     p_weight['23'].T,
                                     u, p1_state, p3_state,
                                     p2_bias)
            p3_state = self._p_state(p_weight['03'],
                                     p_weight['13'],
                                     p_weight['23'],
                                     u, p1_state, p2_state,
                                     p3_bias)

        return u.detach(), p1_state.detach(), \
            p2_state.detach(), p3_state.detach()
