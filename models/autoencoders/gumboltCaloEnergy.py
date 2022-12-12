"""
GumBolt implementation for Calorimeter data
Energy - Add energy related terms to loss function
"""

import typing

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class GumBoltCaloEnergy(GumBoltCaloCRBM):

    def __init__(self, **kwargs):
        super(GumBoltCaloEnergy, self).__init__(**kwargs)
        self.model_type = "GumBoltCaloEnergy"

    def total_energy_loss(self, is_training=True):
        """
        Add a term to the loss function based on the total energy
        """

        logger.debug("GumBoltCaloCRBM::total_energy_loss")

    def loss(self, input_data, fwd_out):
        #Overwritting loss as we have an extra term now. Can this be avoided?

        logger.debug("GumBoltCaloCRBM::loss")

        kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(fwd_out.post_logits, fwd_out.post_samples)
        ae_loss = self._output_loss(input_data, fwd_out.output_activations)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0)

        hit_loss = binary_cross_entropy_with_logits(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.), reduction='none')
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1), dim=0)

        total_energy_loss = self.total_energy_loss()

        return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss, "total_energy_loss": total_energy_loss,
                    "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}
