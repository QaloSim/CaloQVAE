"""
GumBolt implementation for Calorimeter data
CNN - Changed to CNN encoder creation
"""
# Torch imports
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn as nn 

import numpy as np

from models.samplers.GibbsSampling import GS

from utils.stats.condrbm_partition import CRBMStats

from models.autoencoders.gumboltAtlasPRBMCNN import GumBoltAtlasPRBMCNN
from CaloQVAE.models.rbm import pegasusRBM
from CaloQVAE.models.samplers import cpgbs
from models.networks.EncoderUCNN import EncoderUCNN
from models.networks.basicCoders import DecoderCNNCond

import time

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class GumBoltAtlasPCRBMCNN(GumBoltAtlasPRBMCNN):
    """
    GumBolt
    """

    def __init__(self, **kwargs):
        super(GumBoltAtlasPCRBMCNN, self).__init__(**kwargs)
        self._model_type = "GumBoltAtlasPCRBMCNN"
        
        
    def _create_prior(self):
        """Override _create_prior in GumBoltCaloV6.py

        :return: Instance of a PegasusRBM
        """
        assert (self._config.model.n_latent_hierarchy_lvls *
                self._config.model.n_latent_nodes) % 3 == 0, \
            'total no. of latent nodes should be divisible by 3'

        nodes_per_partition = int(self._config.model.n_latent_nodes)
        
        return pegasusRBM.PegasusRBM(nodes_per_partition)
        
    def _create_sampler(self, rbm=None):
        """Override _create_sampler in GumBoltAtlasPRBMCNN.py

        :return: Instance of a CPGBS sampler
        """
        return cpgbs.CPGBS(self.prior, self._config.engine.rbm_batch_size,
                         n_steps=self._config.engine.n_gibbs_sampling_steps)
    
    def _create_stat(self):
        """This object contains methods to compute Stat Mech stuff.

        :return: Instance of a utils.stats.partition.Stats
        """
        return CRBMStats(self.sampler)

    
    def kl_divergence(self, post_logits, post_samples, true_energy, is_training=True):
        """Overrides kl_divergence in GumBoltAtlasPRBMCNN.py

        :param post_logits (list) : List of f(logit_i|x, e) for each hierarchy
                                    layer i. Each element is a tensor of size
                                    (batch_size * n_nodes_per_hierarchy_layer)
        :param post_zetas (list) : List of q(zeta_i|x, e) for each hierarchy
                                   layer i. Each element is a tensor of size
                                   (batch_size * n_nodes_per_hierarchy_layer)
        :param true_energy (list) : List of incidence energies
        """
        # Concatenate all hierarchy levels
        logits_q_z = torch.cat(post_logits, 1)
        post_zetas = torch.cat(post_samples, 1)

        # Compute cross-entropy b/w post_logits and post_samples
        entropy = - self._bce_loss(logits_q_z, post_zetas)
        entropy = torch.mean(torch.sum(entropy, dim=1), dim=0)
        
        # Convert incidence energies into binary representations
        u = self.convert_inc_eng_to_binary(true_energy)

        # Compute positive phase (energy expval under posterior variables) 
        n_nodes_p = self.prior.nodes_per_partition
        pos_energy = self.energy_exp(u, post_zetas[:, :n_nodes_p],
                                     post_zetas[:, n_nodes_p:2*n_nodes_p],
                                     post_zetas[:, 2*n_nodes_p:])
        

        # Compute gradient computation of the logZ term
        u, p1_state, p2_state, p3_state \
            = self.sampler.block_gibbs_sampling(u, post_zetas[:, :n_nodes_p],
                                     post_zetas[:, n_nodes_p:2*n_nodes_p],
                                     post_zetas[:, 2*n_nodes_p:], method=self._config.model.rbmMethod)
        
        neg_energy = - self.energy_exp(u, p1_state, p2_state, p3_state)

        # Estimate of the kl-divergence
        kl_loss = entropy + pos_energy + neg_energy
        return kl_loss, entropy, pos_energy, neg_energy
    
    def energy_exp(self, u, p1_state, p2_state, p3_state):
        """Energy expectation value under the 4-partite CRBM
        Overrides energy_exp in GumBoltAtlasPRBMCNN.py

        :param u        (torch.Tensor) : (batch_size, n_nodes_u)
        :param p1_state (torch.Tensor) : (batch_size, n_nodes_p2)
        :param p2_state (torch.Tensor) : (batch_size, n_nodes_p3)
        :param p3_state (torch.Tensor) : (batch_size, n_nodes_p4)

        :return energy expectation value over the current batch
        """
        w_dict = self.prior.weight_dict
        b_dict = self.prior.bias_dict

        w_dict_cp = {}

        # Broadcast weight matrices (n_nodes_pa, n_nodes_pb) to
        # (batch_size, n_nodes_pa, n_nodes_pb)
        for key in w_dict.keys():
            w_dict_cp[key] = w_dict[key] + torch.zeros((u.size(0),) +
                                                    w_dict[key].size(),
                                                    device=w_dict[key].device)

        # Prepare px_state_t for torch.bmm()
        # Change px_state.size() to (batch_size, 1, n_nodes_px)
        u_t = u.unsqueeze(2).permute(0, 2, 1)
        p1_state_t = p1_state.unsqueeze(2).permute(0, 2, 1)
        p2_state_t = p2_state.unsqueeze(2).permute(0, 2, 1)

        # Prepare py_state for torch.bmm()
        # Change py_state.size() to (batch_size, n_nodes_py, 1)
        p1_state_i = p1_state.unsqueeze(2)
        p2_state_i = p2_state.unsqueeze(2)
        p3_state_i = p3_state.unsqueeze(2)
        

        # Compute the energies for batch samples
        batch_energy = - torch.matmul(p1_state, b_dict['1']) - \
            torch.matmul(p2_state, b_dict['2']) - \
            torch.matmul(p3_state, b_dict['3']) - \
            torch.bmm(u_t,
                      torch.bmm(w_dict_cp['01'], p1_state_i)).reshape(-1) - \
            torch.bmm(u_t,
                      torch.bmm(w_dict_cp['02'], p2_state_i)).reshape(-1) - \
            torch.bmm(u_t,
                      torch.bmm(w_dict_cp['03'], p3_state_i)).reshape(-1) - \
            torch.bmm(p1_state_t,
                      torch.bmm(w_dict_cp['12'], p2_state_i)).reshape(-1) - \
            torch.bmm(p1_state_t,
                      torch.bmm(w_dict_cp['13'], p3_state_i)).reshape(-1) - \
            torch.bmm(p2_state_t,
                      torch.bmm(w_dict_cp['23'], p3_state_i)).reshape(-1)

        return torch.mean(batch_energy, dim=0)
    
    def generate_samples(self, num_samples, true_energy):
        """Generate data samples by decoding CRBM samples

        :param num_samples (int): No. of data samples to generate in one shot
        :param true_energy (int): Incident energy of the particle

        :return true_energies (torch.Tensor): Incident energies of the particle
        for each sample (num_samples,)
        :return samples (torch.Tensor): Data samples, (num_samples, *)
        """
        n_iter = max(num_samples//self.sampler.batch_size, 1)
        true_es, samples = [], []
        
        repeated_energy = true_energy
        if isinstance(true_energy, int):
            repeated_energy = torch.ones((self.sampler.batch_size, 1)) * true_energy
        
        u = self.convert_inc_eng_to_binary(repeated_energy)

        for _ in range(n_iter):
            u, p1_state, p2_state, p3_state = self.sampler.block_gibbs_sampling(u)

            true_e = torch.ones((p1_state.size(0), 1),
                                    device=p1_state.device) * true_energy
            
            prior_samples = torch.cat([p1_state, p2_state, p3_state], dim=1)

            hits, activations = self.decoder(prior_samples, true_e)
            beta = torch.tensor(self._config.model.beta_smoothing_fct,
                                dtype=torch.float, device=hits.device)
            sample = self._inference_energy_activation_fct(activations) \
                * self._hit_smoothing_dist_mod(hits, beta, False)

            true_es.append(true_e)
            samples.append(sample)

        return torch.cat(true_es, dim=0), torch.cat(samples, dim=0)
 
    
    
    def convert_inc_eng_to_binary(self, true_energy):
        
        n_nodes_p = self.prior.nodes_per_partition
        bin_engs = torch.zeros((true_energy.shape[0], n_nodes_p), device=true_energy.device)

        num_int_bits = 20
        num_sqrt_bits = 10
        num_ln_bits = 4
        num_bits = num_int_bits + num_sqrt_bits + num_ln_bits
        
        repeats = n_nodes_p // num_bits
        
        for idx_eng, eng in enumerate(true_energy):
            int_eng = int(eng.item())
            ln_eng = int(np.log(int_eng))
            sqrt_eng = int(np.sqrt(int_eng))
            
            int_bits = bin(int_eng)[2:].zfill(num_int_bits) # Skip '0b', add leading zeros
            sqrt_bits = bin(sqrt_eng)[2:].zfill(num_sqrt_bits)
            ln_bits = bin(ln_eng)[2:].zfill(num_ln_bits)
            
            bits = (ln_bits + sqrt_bits + int_bits) * repeats # repeat
            
            for idx_bit, bit in enumerate(reversed(bits)):
                bin_engs[idx_eng][idx_bit] = int(bit)
        
        return bin_engs
    
    
    
    def loss(self, input_data, fwd_out, true_energy):
        """
        - Overrides loss in GumBoltAtlasCRBMCNN.py
        """
        logger.debug("loss")
        
        # KL Loss
        kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(fwd_out.post_logits, fwd_out.post_samples, true_energy)
        
        # MSE Loss
        sigma = 2 * torch.sqrt(torch.max(input_data, torch.min(input_data[input_data>0])))
        interpolation_param = self._config.model.interpolation_param
        ae_loss = torch.pow((input_data - fwd_out.output_activations)/sigma,2) * (1 - interpolation_param + interpolation_param*torch.pow(sigma,2)) * torch.exp(self._config.model.mse_weight*input_data)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0)
        
        # BCE Hit Loss
        hit_loss = binary_cross_entropy_with_logits(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.), weight = 1+input_data, reduction='none') # weight = torch.sqrt(1+input_data)
        spIdx = torch.where(input_data > 0, 0., 1.).sum(dim=1) / input_data.shape[1]
        sparsity_weight = torch.exp(self._config.model.alpha - self._config.model.gamma * spIdx)
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1) * sparsity_weight, dim=0)
        
        
        return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
                "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy,}
