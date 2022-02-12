"""
GumBolt implementation for Calorimeter data
CRBM - Change the prior and sampler to Chimera RBM
"""
# Python imports
import math
import torch

# DiVAE.models imports
from models.autoencoders.gumboltCaloV6 import GumBoltCaloV6
from models.networks.hierarchicalEncoderV2 import HierarchicalEncoderV2

from models.rbm.chimeraRBM import ChimeraRBM
from models.samplers.pcd import PCD

from DiVAE import logging
logger = logging.getLogger(__name__)

_CELL_SIDE_QUBITS = 4
_MAX_ROW_COLS = 16

class GumBoltCaloCRBM(GumBoltCaloV6):
    
    def __init__(self, **kwargs):
        super(GumBoltCaloCRBM, self).__init__(**kwargs)
        self._model_type = "GumBoltCaloCRBM"
        
        # Initialize the qubit idxs to be used by chains mapping
        # Number of hidden and visible variables on each side of the RBM
        num_var_rbm = (self.n_latent_hierarchy_lvls * self._latent_dimensions)//2
        
        n_cells = math.ceil(num_var_rbm/_CELL_SIDE_QUBITS)
        n_rows = math.ceil(math.sqrt(n_cells))
        n_cols = n_rows
        
        assert n_cols<=_MAX_ROW_COLS
        
        # Idx lists mapping approximate posterior and prior nodes to qubits on the QPU
        visible_qubit_idxs = []
        hidden_qubit_idxs = []
        
        level_1_qubit_idxs = []
        level_2_qubit_idxs = []
        
        # Iterate over the 4x4 chimera cells and assign qubits to latent nodes
        for row in range(n_rows):
            for col in range(n_cols):
                for n in range(_CELL_SIDE_QUBITS):
                    if len(visible_qubit_idxs) < num_var_rbm:
                        idx = 8*row*_MAX_ROW_COLS + 8*col + n
                        
                        # Even cell
                        if (row+col)%2 == 0:
                            visible_qubit_idxs.append(idx)
                            hidden_qubit_idxs.append(idx+4)
                        # Odd cell
                        else:
                            hidden_qubit_idxs.append(idx)
                            visible_qubit_idxs.append(idx+4)
                            
                        # Left side of 4x4 cell always level 1
                        # and right side always level 2
                        level_1_qubit_idxs.append(idx)
                        level_2_qubit_idxs.append(idx+4)
                        
        # Assign the idx list mappings to class variables
        self._visible_qubit_idxs = visible_qubit_idxs
        self._hidden_qubit_idxs = hidden_qubit_idxs
        
        self._level_1_qubit_idxs = level_1_qubit_idxs
        self._level_2_qubit_idxs = level_2_qubit_idxs
                        
    def _create_prior(self):
        """
        - Override _create_prior in discreteVAE.py
        """
        logger.debug("GumBoltCaloCRBM::_create_prior")
        num_rbm_nodes_per_layer=self._config.model.n_latent_hierarchy_lvls*self._latent_dimensions//2
        return ChimeraRBM(n_visible=num_rbm_nodes_per_layer, n_hidden=num_rbm_nodes_per_layer)
 
    def _create_sampler(self, rbm=None):
        """
        - Overrides _create_sampler in discreteVAE.py
        
        Returns:
            PCD Sampler
        """
        logger.debug("GumBoltCaloCRBM::_create_sampler")
        return PCD(batch_size=self._config.engine.rbm_batch_size, RBM=self.prior, n_gibbs_sampling_steps=self._config.engine.n_gibbs_sampling_steps)
    
    def kl_divergence(self, post_logits, post_samples, is_training=True):
        """
        - Compute KLD b.w. hierarchical posterior and RBM prior using GumBolt trick
        - Overrides kl_divergence in gumbolt.py
        - Uses negative energy expectation value as an approximation to logZ
        
        Args:
            post_logits: List of posterior logits (logit_q_z)
            post_samples: List of posterior samples (zeta)
        Returns:
            kl_loss: "Approximate integral KLD" loss whose gradient equals the
                     gradient of the true KLD loss
        """
        logger.debug("GumBoltCaloCRBM::kl_divergence")
        
        # Concatenate all hierarchy levels
        logits_q_z = torch.cat(post_logits, 1)
        post_zetas = torch.cat(post_samples, 1)
        
        # Compute cross-entropy b/w post_logits and post_samples
        entropy = - self._bce_loss(logits_q_z, post_zetas)
        entropy = torch.mean(torch.sum(entropy, 1), 0)
        
        # Compute positive energy expval using hierarchical posterior samples
        
        # Number of hidden and visible variables on each side of the RBM
        num_var_rbm = (self.n_latent_hierarchy_lvls * self._latent_dimensions)//2
        
        # Compute positive energy contribution to the KL divergence
        if "mapping" in self._config.model and self._config.model.mapping.lower()=="chains":
            post_zetas_1, post_zetas_2 = post_zetas[:, :num_var_rbm], post_zetas[:, num_var_rbm:]
            post_zetas_vis, post_zetas_hid = torch.zeros(post_zetas_1.size(), device=post_zetas.device), torch.zeros(post_zetas_1.size(), device=post_zetas.device)
            
            for i, idx in enumerate(self._visible_qubit_idxs):
                if idx in self._level_1_qubit_idxs:
                    post_zetas_vis[:, i] = post_zetas_1[:, i]
                    #print("_visible_qubit_idx : ", idx, " level 1 i : ", i)
                else:
                    post_zetas_vis[:, i] = post_zetas_2[:, i]
                    #print("_visible_qubit_idx : ", idx, " level 2 i : ", i)

            for i, idx in enumerate(self._hidden_qubit_idxs):
                if idx in self._level_1_qubit_idxs:
                    post_zetas_hid[:, i] = post_zetas_1[:, i]
                    #print("_hidden_qubit_idxs : ", idx, " level 1 i : ", i)
                else:
                    post_zetas_hid[:, i] = post_zetas_2[:, i]
                    #print("_hidden_qubit_idxs : ", idx, " level 2 i : ", i)
            
            pos_energy = self.energy_exp(post_zetas_vis, post_zetas_hid)
        else:
            post_zetas_vis, post_zetas_hid = post_zetas[:, :num_var_rbm], post_zetas[:, num_var_rbm:]
            pos_energy = self.energy_exp(post_zetas_vis, post_zetas_hid)
        
        # Compute gradient contribution of the logZ term
        rbm_visible_samples, rbm_hidden_samples = self.sampler.block_gibbs_sampling()
        rbm_vis, rbm_hid = rbm_visible_samples.detach(), rbm_hidden_samples.detach()
        neg_energy = - self.energy_exp(rbm_vis, rbm_hid)
        
        kl_loss = entropy + pos_energy + neg_energy
        return kl_loss, entropy, pos_energy, neg_energy