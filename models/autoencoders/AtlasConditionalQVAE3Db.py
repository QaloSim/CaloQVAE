"""
QVAE for Calorimeter data
It uses either Pegasus or Zephyr
"""
# Torch imports
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn as nn 

import numpy as np

from models.autoencoders.AtlasConditionalQVAE3D import AtlasConditionalQVAE3D


from CaloQVAE import logging
logger = logging.getLogger(__name__)


class AtlasConditionalQVAE3Db(AtlasConditionalQVAE3D):
    """
    GumBolt
    """

    def __init__(self, **kwargs):
        super(AtlasConditionalQVAE3Db, self).__init__(**kwargs)
        

    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 15 * 1.2812657528661318):
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    
    def loss(self, input_data, fwd_out, true_energy):
        """
        - Overrides loss in gumboltCaloV5.py
        """
        logger.debug("loss")

        kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(fwd_out.post_logits, fwd_out.post_samples)  
        
        sigma = 2 * torch.sqrt(torch.max(input_data, torch.min(input_data[input_data>0])))
        interpolation_param = self._config.model.interpolation_param
        batch_mean = torch.mean(input_data[input_data>0.01],dim=0)
        ae_loss = torch.pow((input_data - fwd_out.output_activations)/sigma,2) * (1 - interpolation_param + interpolation_param*torch.pow(sigma,2)) * (torch.exp(self._config.model.pos_mse_weight*(input_data-batch_mean)) + torch.exp(-self._config.model.neg_mse_weight*(input_data-batch_mean)))

        # Reweight the events with the incidental energy in AE loss.
        if self._config.model.weighted_ae_loss:
            true_energy_weight = self.trans_energy(true_energy).squeeze(dim=1)
            ae_loss = torch.mean(torch.dot(torch.sum(ae_loss, dim=1) , true_energy_weight), dim=0) * self._config.model.coefficient
        else:
            ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0) * self._config.model.coefficient
        
        hit_loss = binary_cross_entropy_with_logits(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.), weight= (1+input_data).pow(self._config.model.bce_weights_power), reduction='none') #, weight= 1 + input_data: (1+input_data).sqrt()
        spIdx = torch.where(input_data > 0, 0., 1.).sum(dim=1) / input_data.shape[1]
        sparsity_weight = torch.exp(self._config.model.alpha - self._config.model.gamma * spIdx)
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1) * sparsity_weight, dim=0)

        return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
                "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}