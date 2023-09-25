"""
GumBolt implementation for Calorimeter data
CNN - Changed to CNN encoder creation
"""
# Torch imports
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn as nn 

from models.samplers.GibbsSampling import GS

# DiVAE.models imports
from models.autoencoders.gumboltAtlasCRBMCNN import GumBoltAtlasCRBMCNN
from models.networks.EncoderCNN import EncoderCNN
from models.networks.EncoderUCNN import EncoderUCNN
from models.networks.basicCoders import DecoderCNN, Classifier, DecoderCNNCond, DecoderCNNCondSmall

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class GumBoltAtlasCRBMCNNDCond(GumBoltAtlasCRBMCNN):
    """
    GumBolt
    """

    def __init__(self, **kwargs):
        super(GumBoltAtlasCRBMCNNDCond, self).__init__(**kwargs)
        self._model_type = "GumBoltAtlasCRBMCNNDCond"
        self._bce_loss = BCEWithLogitsLoss(reduction="none")

    
    def _create_decoder(self):
        """
        - Overrides _create_decoder in gumboltAtlasCRBMCNN.py

        Returns:
            DecoderCNNCond instance
        """
        logger.debug("GumBoltAtlasCRBMCNN::_create_decoder")
        self._decoder_nodes[0] = (self._decoder_nodes[0][0]+1,
                                  self._decoder_nodes[0][1])
        # return DecoderCNNCond(node_sequence=self._decoder_nodes,
        #                       activation_fct=self._activation_fct, #<--- try identity
        #                       num_output_nodes = self._flat_input_size,
        #                       cfg=self._config)
        return DecoderCNNCondSmall(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct, #<--- try identity
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)




