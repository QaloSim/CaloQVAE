"""
Default Engine Class for various autoencoder models.

Tested with:
- Autoencoder
"""

import torch
import os
import coffea

# Weights and Biases
import wandb
import numpy as np

from engine.engineAtlas import EngineAtlas

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class EngineAtlasV2(EngineAtlas):

    def __init__(self, cfg, **kwargs):
        logger.info("Setting up engine AtlaV2s.")
        super(EngineAtlasV2, self).__init__(cfg, **kwargs)
        
    def _reduce(self, in_data, true_energy, R=1e-7):
        """
        Hao Transformation Scheme
        """
        
        return torch.log1p(in_data)

        
    def _reduceinv(self, in_data, true_energy, R=1e-7):
        """
        Hao Transformation Scheme
        """
        
        return in_data.exp() - 1
        
if __name__=="__main__":
    logger.info("Willkommen!")
    engine=Engine()
    logger.info("Success!")