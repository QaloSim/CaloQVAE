"""
GumBolt implementation for Calorimeter data
CRBM - Change the prior and sampler to Chimera RBM
"""
# Python imports
import math
import torch
import numpy as np

# DiVAE.models imports
from models.autoencoders.gumboltCaloV6 import GumBoltCaloV6

from models.rbm.chimeraRBM import ChimeraRBM
from models.rbm.chimerav2 import QimeraRBM
# from models.samplers.pcd import PCD
from models.samplers.GibbsSampling import GS
from models.networks.hierarchicalEncoderV2 import HierarchicalEncoderV2
from models.networks.basicCoders import BasicDecoderV3

from dwave.system import DWaveSampler
from notebooks.nbutils import *

from CaloQVAE import logging
logger = logging.getLogger(__name__)

_CELL_SIDE_QUBITS = 4
_MAX_ROW_COLS = 16

class GumBoltCaloCRBMUnCond(GumBoltCaloV6):
    
    def __init__(self, **kwargs):
        super(GumBoltCaloCRBMUnCond, self).__init__(**kwargs)
        self._model_type = "GumBoltCaloCRBMUnCond"
        
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
        
        # Initialize the DWave QPU sampler
#         self._qpu_sampler = DWaveSampler(solver={"topology__type":"chimera", "chip_id":"DW_2000Q_6"})
        self._qpu_sampler = DWaveSampler(solver={"topology__type":"pegasus"})

    def _create_prior(self):
        """
        - Override _create_prior in discreteVAE.py
        """
        logger.debug("GumBoltCaloCRBMUnCond::_create_prior")
        num_rbm_nodes_per_layer=self._config.model.n_latent_hierarchy_lvls*self._latent_dimensions//2
        return QimeraRBM(n_visible=num_rbm_nodes_per_layer, n_hidden=num_rbm_nodes_per_layer, fullyconnected=self._config.model.fullyconnected)
        
    def _create_sampler(self, rbm=None):
        """
        - Overrides _create_sampler in discreteVAE.py
        
        Returns:
            Gibbs Sampler
        """
        logger.debug("GumBoltCaloCRBMUnCond::_create_sampler")
        return GS(batch_size=self._config.engine.rbm_batch_size,
                   RBM=self.prior,
                   n_gibbs_sampling_steps\
                       =self._config.engine.n_gibbs_sampling_steps)
    
    def _create_encoder(self):
        """
        - Overrides _create_encoder in GumBoltCaloV5.py

        Returns:
            Hierarchical Encoder instance
        """
        logger.debug("GumBoltCaloCRBMUnCond::_create_encoder")
        return HierarchicalEncoderV2(
            input_dimension=self._flat_input_size,
            n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
            n_latent_nodes=self.n_latent_nodes,
            skip_latent_layer=False,
            smoother="Gumbel",
            cfg=self._config)
    
    def _create_decoder(self):
        logger.debug("GumBoltCaloCRBMUnCond::_create_decoder")
        return BasicDecoderV3(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct,
                              cfg=self._config)
    
    def forward(self, x, is_training, beta_smoothing_fct=5, slope_act_fct=0.02):
        """
        - Overrides forward in dvaepp.py
        
        Returns:
            out: output container 
        """
        logger.debug("forward")
        
        #see definition for explanation
        out=self._output_container.clear()
        
	    #Step 1: Feed data through encoder
        in_data = x[0] #torch.cat([x[0], x[1]], dim=1)
        out.beta, out.post_logits, out.post_samples = self.encoder(in_data, is_training, beta_smoothing_fct)
        post_samples = torch.cat(out.post_samples, 1)
        
        output_hits, output_activations = self.decoder(post_samples)
        
        out.output_hits = output_hits
        beta = torch.tensor(self._config.model.output_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
        out.output_activations = self._energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
            
        return out
    
    
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
        num_var_rbm = (self.n_latent_hierarchy_lvls 
                       * self._latent_dimensions)//2
        
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
    
    
    def generate_samples(self, num_samples=64, true_energy=None):
        """
        generate_samples()
        """
        true_energies = []
        num_iterations = max(num_samples//self.sampler.get_batch_size(), 1)
        samples = []
        for i in range(num_iterations):
            rbm_visible_samples, rbm_hidden_samples = self.sampler.block_gibbs_sampling()
            rbm_vis = rbm_visible_samples.detach()
            rbm_hid = rbm_hidden_samples.detach()
            
            if true_energy is None:
                true_e = torch.rand((rbm_vis.size(0), 1), device=rbm_vis.device).detach() * 100.
            else:
                true_e = torch.ones((rbm_vis.size(0), 1), device=rbm_vis.device).detach() * true_energy
                
            prior_samples = torch.cat([rbm_vis, rbm_hid], dim=1)
            
            output_hits, output_activations = self.decoder(prior_samples)
            beta = torch.tensor(self._config.model.beta_smoothing_fct,
                                dtype=torch.float, device=output_hits.device,
                                requires_grad=False)
            sample = self._energy_activation_fct(output_activations) \
                * self._hit_smoothing_dist_mod(output_hits, beta, False)
            
            true_energies.append(true_e) 
            samples.append(sample)
            
        return torch.cat(true_energies, dim=0), torch.cat(samples, dim=0)
    
    
