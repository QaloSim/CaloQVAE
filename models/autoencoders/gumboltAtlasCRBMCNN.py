"""
GumBolt implementation for Calorimeter data
CNN - Changed to CNN encoder creation
"""
# Torch imports
import torch

# DiVAE.models imports
from models.autoencoders.gumboltCaloCRBM import GumBoltCaloCRBM
from models.networks.EncoderCNN import EncoderCNN
from models.networks.basicCoders import DecoderCNN

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
            input_dimension=self._flat_input_size,
            n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
            n_latent_nodes=self.n_latent_nodes,
            skip_latent_layer=False,
            smoother="Gumbel",
            cfg=self._config)
    
    def _create_decoder(self):
        """
        - Overrides _create_decoder in GumBoltCaloV5.py

        Returns:
            DecoderCNN instance
        """
        logger.debug("GumBoltAtlasCRBMCNN::_create_decoder")
        self._decoder_nodes[0] = (self._decoder_nodes[0][0]+1,
                                  self._decoder_nodes[0][1])
        return DecoderCNN(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct, #<--- try identity
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)
    
    def forward(self, xx, is_training):
        """
        - Overrides forward in GumBoltCaloV5.py
        
        Returns:
            out: output container 
        """
        logger.debug("forward")
        
        #see definition for explanation
        out=self._output_container.clear()
        x, x0 = xx
        
	    #Step 1: Feed data through encoder
        # in_data = torch.cat([x[0], x[1]], dim=1)
        
        out.beta, out.post_logits, out.post_samples = self.encoder(x, x0, is_training)
        # out.post_samples = self.encoder(x, x0, is_training)
        post_samples = out.post_samples
        post_samples = torch.cat(out.post_samples, 1)
#         post_samples = torch.cat([post_samples, x[1]], dim=1)
        
        output_hits, output_activations = self.decoder(post_samples, x0)
        
        out.output_hits = output_hits
        beta = torch.tensor(self._config.model.output_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
        out.output_activations = self._energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
        return out
    
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
#             prior_samples = torch.cat([rbm_vis, rbm_hid, true_e], dim=1)
            prior_samples = torch.cat([rbm_vis, rbm_hid], dim=1)
            
            output_hits, output_activations = self.decoder(prior_samples, true_e)
            beta = torch.tensor(self._config.model.beta_smoothing_fct,
                                dtype=torch.float, device=output_hits.device,
                                requires_grad=False)
            sample = self._energy_activation_fct(output_activations) \
                * self._hit_smoothing_dist_mod(output_hits, beta, False)
            
            true_energies.append(true_e) 
            samples.append(sample)
            
        return torch.cat(true_energies, dim=0), torch.cat(samples, dim=0)
