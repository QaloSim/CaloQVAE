"""
GumBolt implementation for Calorimeter data
CNN - Changed to CNN encoder creation
"""

# DiVAE.models imports
from models.autoencoders.gumboltCaloCRBM import GumBoltCaloCRBM
from models.networks.EncoderCNN import EncoderCNN

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class GumBoltAtlasCRBMCNN(GumBoltCaloCRBM):
    """
    GumBolt
    """

    def __init__(self, **kwargs):
        super(GumBoltAtlasCRBMCNN, self).__init__(**kwargs)
        self._model_type = "GumBoltAtlasCRBMCNN"

    def _create_encoder(self):
        """
        - Overrides _create_encoder in GumBoltCaloCRBM.py

        Returns:
            EncoderCNN instance
        """
        logger.debug("GumBoltAtlasCRBMCNN::_create_encoder")
        return EncoderCNN(
            input_dimension=self._flat_input_size+1,
            n_latent_hierarchy_lvls=1, #engine.model.n_latent_hierarchy_lvls
            n_latent_nodes=self.n_latent_nodes,
            skip_latent_layer=False,
            smoother="Gumbel",
            cfg=self._config)
