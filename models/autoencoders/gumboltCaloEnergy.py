"""
GumBolt implementation for Calorimeter data
Energy - Add energy related terms to loss function
"""

import torch

# DiVAE.models imports
from models.autoencoders.gumboltCaloCRBM import GumBoltCaloCRBM

from CaloQVAE import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GumBoltCaloEnergy(GumBoltCaloCRBM):

    def __init__(self, **kwargs):
        super(GumBoltCaloEnergy, self).__init__(**kwargs)
        self.model_type = "GumBoltCaloEnergy"

    def total_energy_loss(self, input_data, fwd_output, is_training=True):
        """
        Add a term to the loss function based on the total energy
        """

        layer_split = [288, 144, 72] #Cell count per layer
        logger.debug("GumBoltCaloEnergy::total_energy_loss")
        total_energy = torch.sum(fwd_output, dim=1)
        in_data = torch.sum(input_data, dim=1)
        layers = torch.split(input_data, layer_split, dim=1)
        summed_layers = [torch.sum(layer, dim=1) for layer in layers]
        for i in range(len(input_data)):
            logger.debug(f"In data: {in_data[i]}, layer0: {summed_layers[0][i]}, layer1: {summed_layers[1][i]}, layer2: {summed_layers[2][i]}")

        loss = torch.nn.L1Loss()
        total_energy_loss = loss(total_energy, in_data)

        return total_energy_loss
