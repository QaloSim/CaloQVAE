"""
Pegasus PCD (4-partite block Gibbs Sampler)

Author : Abhi (abhishek@myumanitoba.ca)
"""

from torch import nn

class PPCD(nn.Module):
    """
    Pegasus PCD (4-partite block Gibbs) sampler

    Attributes:
        _prbm:
        _batch_size: 
    """

    def __init__(self, batch_size, PRBM, **kwargs):
        """__init__()

        :param batch_size (int) : No. of blocked Gibbs chains to run in parallel
        :param PRBM (object)    : Configuration of the Pegasus RBM

        :return None
        """
        super().__init__(**kwargs)
        self._batch_size = batch_size
        self._prbm = PRBM

    def block_gibbs_sampling(self):
        """block_gibbs_sampling()

        :param 

        :return samples
        """
