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
from models.networks.EncoderUCNN import EncoderUCNNH
from models.networks.basicCoders import DecoderCNNUnCondSmall

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class GumBoltAtlasCRBMCNNUnCond(GumBoltAtlasCRBMCNN):
    """
    GumBolt
    """

    def __init__(self, **kwargs):
        super(GumBoltAtlasCRBMCNNUnCond, self).__init__(**kwargs)
        self._model_type = "GumBoltAtlasCRBMCNNUnCond"
        self._bce_loss = BCEWithLogitsLoss(reduction="none")
        
    def _create_encoder(self):
        """
        - Overrides _create_encoder in GumBoltCaloCRBM.py

        Returns:
            EncoderUCNNH instance
        """
        logger.debug("GumBoltAtlasCRBMCNN::_create_encoder")
        
        return EncoderUCNNH(encArch='UncondSmall',
            input_dimension=self._flat_input_size,
            n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
            n_latent_nodes=self.n_latent_nodes,
            skip_latent_layer=False,
            smoother="Gumbel",
            cfg=self._config)

    
    def _create_decoder(self):
        """
        - Overrides _create_decoder in gumboltAtlasCRBMCNN.py

        Returns:
            DecoderCNNCond instance
        """
        logger.debug("GumBoltAtlasCRBMCNN::_create_decoder")
        self._decoder_nodes[0] = (self._decoder_nodes[0][0]+1,
                                  self._decoder_nodes[0][1])
        
        return DecoderCNNUnCondSmall(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct, #<--- try identity
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)




