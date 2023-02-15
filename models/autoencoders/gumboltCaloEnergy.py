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
        logger.debug("GumBoltCaloEnergy::total_energy_loss")
        assert(input_data.shape == fwd_output.shape)

        total_energy = torch.sum(fwd_output, dim = 1)
        in_data = torch.sum(input_data, dim = 1)

        loss = torch.nn.L1Loss()
        total_energy_loss = loss(total_energy, in_data)

        return total_energy_loss

    def layer_energy_loss(self, input_data, fwd_output, is_training=True):
        """
        Add a term to the loss funtion based on the energy per calo layer
        """

        logger.debug("GumBoltCaloEnergy::layer_energy_loss")
        assert(input_data.shape == fwd_output.shape)

        layer_split = [288, 144, 72] #Cell count per layer

        data_layers = torch.split(input_data, layer_split, dim = 1)
        reco_layers = torch.split(fwd_output, layer_split, dim = 1)
        data_summed_layers = [torch.sum(layer, dim = 1) for layer in data_layers]
        reco_summed_layers = [torch.sum(layer, dim = 1) for layer in reco_layers]
        data_summed_total = torch.sum(input_data, dim = 1)
        reco_summed_total = torch.sum(fwd_output, dim = 1)

        loss_list = []

        for i in range(len(layer_split)):
            loss = torch.nn.L1Loss()
            energy_loss = loss(torch.div(reco_summed_layers[i], reco_summed_total), torch.div(data_summed_layers[i], data_summed_total))
            loss_list.append(energy_loss)

        return loss_list
