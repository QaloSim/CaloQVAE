"""
GumBolt implementation for Calorimeter data
V5 - Energy conditioning added to the decoder - Energy conditioning added to the encoder

Author : Abhi (abhishek@myumanitoba.ca)
"""

# Torch imports
import torch
from torch.nn import ReLU, MSELoss, BCEWithLogitsLoss, L1Loss, Sigmoid
from torch.nn.functional import binary_cross_entropy_with_logits

# DiVAE.models imports
from models.autoencoders.gumbolt import GumBolt
from models.networks.basicCoders import BasicDecoderV3
from models.networks.hierarchicalEncoder import HierarchicalEncoder

# DiVAE.utils imports
from utils.dists.gumbelmod import GumbelMod

from CaloQVAE_unknown import logging
logger = logging.getLogger(__name__)
import json
import time

class GumBoltCaloV5(GumBolt):
    
    def __init__(self, **kwargs):
        super(GumBoltCaloV5, self).__init__(**kwargs)
        self._model_type = "GumBoltCaloV5"
        self._energy_activation_fct = ReLU()
        self._hit_activation_fct = Sigmoid()
        self._output_loss = MSELoss(reduction="none")
        self._hit_loss = BCEWithLogitsLoss(reduction="none")
        
        self._hit_smoothing_dist_mod = GumbelMod()
        
    def forward(self, x, is_training):
        """
        - Overrides forward in dvaepp.py
        
        Returns:
            out: output container 
        """
        logger.debug("forward")
        
        #see definition for explanation
        out=self._output_container.clear()
        
	    #Step 1: Feed data through encoder
        in_data = torch.cat([x[0], x[1]], dim=1)
        out.beta, out.post_logits, out.post_samples = self.encoder(in_data, is_training)
        post_samples = torch.cat(out.post_samples, 1)
        post_samples = torch.cat([post_samples, x[1]], dim=1)
        
        output_hits, output_activations = self.decoder(post_samples)
        
        out.output_hits = output_hits
        beta = torch.tensor(self._config.model.output_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
        out.output_activations = self._energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
        return out
    
    def loss(self, input_data, fwd_out):
        logger.debug("loss")
        
        kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(fwd_out.post_logits, fwd_out.post_samples)
        ae_loss = self._output_loss(input_data, fwd_out.output_activations)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0)
        
        #hit_loss = self._hit_loss(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.))
        #hit_loss = torch.mean(torch.sum(hit_loss, dim=1), dim=0)
        hit_loss = binary_cross_entropy_with_logits(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.), reduction='none')
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1), dim=0)
        
        return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
                "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}
    
    # def generate_samples(self, num_samples=64, true_energy=None):
    #     """
    #     generate_samples()
    #     """
    #     true_energies = []
    #     num_iterations = max(num_samples//self.sampler.get_batch_size(), 1)
    #     samples = []
    #     for i in range(num_iterations):
    #         rbm_visible_samples, rbm_hidden_samples = self.sampler.block_gibbs_sampling()
    #         rbm_vis = rbm_visible_samples.detach()
    #         rbm_hid = rbm_hidden_samples.detach()
            
    #         if true_energy is None:
    #             true_e = torch.rand((rbm_vis.size(0), 1), device=rbm_vis.device).detach() * 100.
    #         else:
    #             true_e = torch.ones((rbm_vis.size(0), 1), device=rbm_vis.device).detach() * true_energy[0] + torch.rand((rbm_vis.size(0), 1), device=rbm_vis.device).detach() * (true_energy[1]-true_energy[0])

    #         prior_samples = torch.cat([rbm_vis, rbm_hid, true_e], dim=1)
            
    #         output_hits, output_activations = self.decoder(prior_samples)
    #         beta = torch.tensor(self._config.model.beta_smoothing_fct,
    #                             dtype=torch.float, device=output_hits.device,
    #                             requires_grad=False)
    #         sample = self._energy_activation_fct(output_activations) \
    #             * self._hit_smoothing_dist_mod(output_hits, beta, False)
            
    #         true_energies.append(true_e) 
    #         samples.append(sample)
    #     return torch.cat(true_energies, dim=0), torch.cat(samples, dim=0)




    def generate_samples(self, num_samples=64, true_energy=None):
        """
        generate_samples()
        """
        true_energies = []
        num_iterations = max(num_samples//self.sampler.get_batch_size(), 1)
        samples = []
        
        block_gibbs_sampling_times = []
        decoder_times = []
        energy_activation_fct_times = []
        hit_smoothing_dist_mod_times = []
        batch_loading_times = []
        previous_data = {}
        # Load the previous data from the JSON file
        try:
            with open('time_monitoring.json', 'r') as f:
                previous_data = json.load(f)
                # block_gibbs_sampling_times = previous_data['block_gibbs_sampling_times']
                # decoder_times = previous_data['decoder_times']
                # energy_activation_fct_times = previous_data['energy_activation_fct_times']
                # hit_smoothing_dist_mod_times = previous_data['hit_smoothing_dist_mod_times']
                # total_time = previous_data['total_time']  # Load previous total_time
                # loading_time = previous_data['loading_time']  # Load previous loading_time
        except FileNotFoundError:
            pass
        
        start_sampling_time = time.time()  # Record the start time for sampling
        
        for i in range(num_iterations):
            block_gibbs_start_time = time.time()  # Record the start time for block_gibbs_sampling
            rbm_visible_samples, rbm_hidden_samples = self.sampler.block_gibbs_sampling()
            rbm_vis = rbm_visible_samples.detach()
            rbm_hid = rbm_hidden_samples.detach()
            block_gibbs_end_time = time.time()  # Record the end time for block_gibbs_sampling
            block_gibbs_time = block_gibbs_end_time - block_gibbs_start_time
            block_gibbs_sampling_times.append(block_gibbs_time)
            
            if true_energy is None:
                true_e = torch.rand((rbm_vis.size(0), 1), device=rbm_vis.device).detach() * 100.
            else:
                true_e = true_e = torch.ones((rbm_vis.size(0), 1), device=rbm_vis.device).detach() * true_energy[0] + torch.rand((rbm_vis.size(0), 1), device=rbm_vis.device).detach() * (true_energy[1]-true_energy[0])
            start_loading_time = time.time()  # Record the start time for loading to GPU
            prior_samples = torch.cat([rbm_vis, rbm_hid, true_e], dim=1)
            end_loading_time = time.time()  # Record the end time for loading to GPU
            batch_loading_time = end_loading_time - start_loading_time  # Calculate the loading time
            batch_loading_times.append(batch_loading_time)
            decoder_start_time = time.time()  # Record the start time for decoder
            output_hits, output_activations = self.decoder(prior_samples)
            decoder_end_time = time.time()  # Record the end time for decoder
            decoder_time = decoder_end_time - decoder_start_time
            decoder_times.append(decoder_time)
            
            energy_activation_fct_start_time = time.time()  # Record the start time for _energy_activation_fct
            sample = self._energy_activation_fct(output_activations)
            energy_activation_fct_end_time = time.time()  # Record the end time for _energy_activation_fct
            energy_activation_fct_time = energy_activation_fct_end_time - energy_activation_fct_start_time
            energy_activation_fct_times.append(energy_activation_fct_time)
            beta = torch.tensor(self._config.model.beta_smoothing_fct,
                                dtype=torch.float, device=output_hits.device,
                                requires_grad=False)
            hit_smoothing_dist_mod_start_time = time.time()  # Record the start time for _hit_smoothing_dist_mod
            sample *= self._hit_smoothing_dist_mod(output_hits, beta, False)
            hit_smoothing_dist_mod_end_time = time.time()  # Record the end time for _hit_smoothing_dist_mod
            hit_smoothing_dist_mod_time = hit_smoothing_dist_mod_end_time - hit_smoothing_dist_mod_start_time
            hit_smoothing_dist_mod_times.append(hit_smoothing_dist_mod_time)
            
            true_energies.append(true_e) 
            samples.append(sample)
        
        end_sampling_time = time.time()  # Record the end time for sampling
        
        total_block_gibbs_time = sum(block_gibbs_sampling_times)
        total_decoder_time = sum(decoder_times)
        total_energy_activation_fct_time = sum(energy_activation_fct_times)
        total_hit_smoothing_dist_mod_time = sum(hit_smoothing_dist_mod_times)
        loading_time = sum(batch_loading_times)
        sampling_time = end_sampling_time - start_sampling_time
        
        true_energies_tensor = torch.cat(true_energies, dim=0)
        samples_tensor = torch.cat(samples, dim=0)
        
        # Build the updated time monitoring data dictionary
        data = {
            'sample_size':[num_samples],
            'sampling_time': [sampling_time],
            'block_gibbs_sampling_times': [total_block_gibbs_time],
            'decoder_times': [total_decoder_time],
            'energy_activation_fct_times': [total_energy_activation_fct_time],
            'hit_smoothing_dist_mod_times': [total_hit_smoothing_dist_mod_time],
            'loading_time': [loading_time]
        }
        
        # Merge the previous data with the new data
        if 'block_gibbs_sampling_times' in previous_data:
            previous_data['sample_size'].append(num_samples)
            previous_data['sampling_time'].append(sampling_time)
            previous_data['block_gibbs_sampling_times'].append(total_block_gibbs_time)
            previous_data['decoder_times'].append(total_decoder_time)
            previous_data['energy_activation_fct_times'].append(total_energy_activation_fct_time)
            previous_data['hit_smoothing_dist_mod_times'].append(total_hit_smoothing_dist_mod_time)
            previous_data['loading_time'].append(loading_time) 
            data = previous_data
        # Only save the timing data when the number of samples is larger than 255.
        if num_samples > 255:
            # Save the updated data back to the JSON file
            with open('time_monitoring.json', 'w') as f:
                json.dump(data, f, indent=2)
            
        return true_energies_tensor, samples_tensor

    
    def _create_decoder(self):
        logger.debug("GumBoltCaloV5::_create_decoder")
        self._decoder_nodes[0] = (self._decoder_nodes[0][0]+1,
                                  self._decoder_nodes[0][1])
        return BasicDecoderV3(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct,
                              cfg=self._config)
    
    def _create_encoder(self):
        """
        - Overrides _create_encoder in gumbolt.py
        
        Returns:
            Hierarchical Encoder instance
        """
        logger.debug("GumBoltCaloV5::_create_encoder")
        return HierarchicalEncoder(
            input_dimension=self._flat_input_size+1,
            n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
            n_latent_nodes=self.n_latent_nodes,
            n_encoder_layer_nodes=self.n_encoder_layer_nodes,
            n_encoder_layers=self.n_encoder_layers,
            skip_latent_layer=False,
            smoother="Gumbel",
            cfg=self._config)
