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

from models.samplers.GibbsSampling import GS

# from utils.stats.partition import Stats
from utils.stats.cond_partition import Stats
# from utils.flux_biases import h_to_fluxbias #This should change to dwave's repo
from dwave.system.temperatures import h_to_fluxbias

# DiVAE.models imports
# from models.autoencoders.gumboltAtlasCRBMCNNDecCond import GumBoltAtlasCRBMCNNDCond
from models.autoencoders.gumboltAtlasPRBMCNN import GumBoltAtlasPRBMCNN
from models.autoencoders.AtlasConditionalQVAE import AtlasConditionalQVAE
from CaloQVAE.models.rbm import pegasusRBM, zephyrRBM
from CaloQVAE.models.samplers import pgbs

from models.networks.EncoderCond import EncoderHierarchyPB_BinEv2
from models.networks.DecoderCond import DecoderCNNPB, DecoderCNNPBv2, DecoderCNNPBv3, DecoderCNNPBv4, DecoderCNNPBv4_HEMOD, DecoderCNNPB_HEv1, DecoderCNNPB3Dv1, DecoderCNNPB3Dv2, DecoderCNNPB3Dv3, DecoderCNNPB3Dv4

import time

from CaloQVAE import logging
logger = logging.getLogger(__name__)


class AtlasConditionalQVAE3D(AtlasConditionalQVAE): #(GumBoltAtlasPRBMCNN): #AtlasConditionalQVAE
    """
    GumBolt
    """

    def __init__(self, **kwargs):
        super(AtlasConditionalQVAE3D, self).__init__(**kwargs)
        self._model_type = "AtlasConditionalQVAE3D"
        self._bce_loss = BCEWithLogitsLoss(reduction="none")
        
    def _create_prior(self):
        """Override _create_prior in GumBoltCaloV6.py

        :return: Instance of a PegasusRBM
        """
        
        nodes_per_partition = self._config.model.n_latent_nodes_per_p
        
        if 'topology' in self._config.model and self._config.model.topology == 'pegasus':
            return pegasusRBM.PegasusRBM(nodes_per_partition, True, self._config.model.fullyconnected)
        elif 'topology' in self._config.model and self._config.model.topology == 'zephyr':
            return zephyrRBM.ZephyrRBM(nodes_per_partition, self._config.model.fullyconnected)
        else:
            return pegasusRBM.PegasusRBM(nodes_per_partition, True, self._config.model.fullyconnected)
        
    def _create_sampler(self, rbm=None):
        """Override _create_sampler in GumBoltCaloV6.py

        :return: Instance of a PGBS sampler
        """
        return pgbs.PGBS(self.prior, self._config.engine.rbm_batch_size,
                         n_steps=self._config.engine.n_gibbs_sampling_steps)
    
    def _create_stat(self):
        """This object contains methods to compute Stat Mech stuff.

        :return: Instance of a utils.stats.partition.Stats
        """
        return Stats(self.sampler)
    
    def create_networks(self):
        logger.debug("Creating Network Structures")
        self.encoder=self._create_encoder()
        self.prior=self._create_prior()
        self.decoder=self._create_decoder()
        self.sampler = self._create_sampler(rbm=self.prior)
        self.stater = self._create_stat()
        
        self._qpu_sampler = self.prior._qpu_sampler
        self.sampling_time_qpu = []
        self.sampling_time_gpu = []
        
    def _create_encoder(self):
        """
        - Overrides _create_encoder in GumBoltCaloCRBM.py

        Returns:
            EncoderCNN instance
        """
        logger.debug("AtlasConditionalQVAE::_create_encoder")
        
        # return EncoderHierarchyPB_BinE(encArch=self._config.model.encodertype,
        #     input_dimension=self._flat_input_size,
        #     n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
        #     n_latent_nodes=self.n_latent_nodes,
        #     skip_latent_layer=False,
        #     smoother="Gumbel",
        #     cfg=self._config)
        return EncoderHierarchyPB_BinEv2(encArch=self._config.model.encodertype,
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
            DecoderCNNCond instance
        """
        logger.debug("GumBoltAtlasCRBMCNN::_create_decoder")
        logger.info(f'GumBoltAtlasCRBMCNN::decoder {self._config.model.decodertype}')
        self._decoder_nodes[0] = (self._decoder_nodes[0][0]+1,
                                  self._decoder_nodes[0][1])

        if self._config.model.decodertype == "SmallPB":
            return DecoderCNNPB(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct,
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)
        elif self._config.model.decodertype == "SmallPBv2":
            return DecoderCNNPBv2(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct,
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)
        elif self._config.model.decodertype == "SmallPBv4":
            return DecoderCNNPBv4(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct,
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)
        elif self._config.model.decodertype == "SmallPBHEv1":
            return DecoderCNNPB_HEv1(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct,
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)
        elif self._config.model.decodertype == "SmallPB3Dv1":
            return DecoderCNNPB3Dv1(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct,
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)
        elif self._config.model.decodertype == "SmallPB3Dv2":
            return DecoderCNNPB3Dv2(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct,
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)
        elif self._config.model.decodertype == "SmallPB3Dv3":
            return DecoderCNNPB3Dv3(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct,
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)
        elif self._config.model.decodertype == "SmallPB3Dv4":
            return DecoderCNNPB3Dv4(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct,
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)


#     def forward(self, xx, is_training, beta_smoothing_fct=5, act_fct_slope=0.02):
#         """
#         - Overrides forward in GumBoltAtlasCRBMCNN.py

#         Returns:
#             out: output container 
#         """
#         logger.debug("forward")

#         #see definition for explanation
#         out=self._output_container.clear()
#         x, x0 = xx

#         self.x_raw = x
#         self.act_fct_slope = act_fct_slope
# 	    #Step 1: Feed data through encoder
#         # in_data = torch.cat([x[0], x[1]], dim=1)

#         out.beta, out.post_logits, out.post_samples = self.encoder(x, x0, is_training, beta_smoothing_fct)
#         # out.post_samples = self.encoder(x, x0, is_training)
#         post_samples = out.post_samples
#         post_samples = torch.cat(out.post_samples, 1)
# #         post_samples = torch.cat([post_samples, x[1]], dim=1)

#         output_hits, output_activations = self.decoder(post_samples, x0, act_fct_slope, x)
#         # labels = self.classifier(output_hits)

#         out.output_hits = output_hits

#         beta = torch.tensor(self._config.model.output_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
#         # if self._config.engine.modelhits:
#         if self.training:
#             # out.output_activations = self._energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
#             activation_fct_annealed = self._training_activation_fct(act_fct_slope)
#             # out.output_activations = activation_fct_annealed(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
#             out.output_activations = activation_fct_annealed(output_activations) * torch.where(x > 0, 1., 0.)
#         else:
#             out.output_activations = self._inference_energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
#         return out

#     def loss(self, input_data, fwd_out, true_energy):
#         """
#         - Overrides loss in gumboltCRBMCNN.py
#         """
#         logger.debug("loss")
#         kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(fwd_out.post_logits, fwd_out.post_samples)
#         # ae_loss = self._output_loss(input_data, fwd_out.output_activations) * torch.exp(self._config.model.mse_weight*input_data)
#         sigma = 2 * torch.sqrt(torch.max(input_data, torch.min(input_data[input_data>0])))
#         interpolation_param = self._config.model.interpolation_param
#         ae_loss = torch.pow((input_data - fwd_out.output_activations)/sigma,2) * (1 - interpolation_param + interpolation_param*torch.pow(sigma,2)) * torch.exp(self._config.model.mse_weight*input_data)
#         ae_loss = torch.mean(torch.mean(ae_loss, dim=1), dim=0)

#         #hit_loss = self._hit_loss(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.))
#         #hit_loss = torch.mean(torch.sum(hit_loss, dim=1), dim=0)
#         # hit_loss = binary_cross_entropy_with_logits(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.), reduction='none')
#         hit_loss = binary_cross_entropy_with_logits(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.), weight= (1+input_data).pow(self._config.model.bce_weights_power), reduction='none') #, weight= 1 + input_data: (1+input_data).sqrt()
#         spIdx = torch.where(input_data > 0, 0., 1.).sum(dim=1) / input_data.shape[1]
#         sparsity_weight = torch.exp(self._config.model.alpha - self._config.model.gamma * spIdx)
#         hit_loss = torch.mean(torch.sum(hit_loss, dim=1) * sparsity_weight, dim=0)

#         # labels_target = nn.functional.one_hot(true_energy.divide(256).log2().to(torch.int64), num_classes=15).squeeze(1).to(torch.float)
#         # hit_label = binary_cross_entropy_with_logits(fwd_out.labels, labels_target)


#         # if self._config.engine.modelhits:
#         return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
#                 "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}
#         # else:
#         #     return {"ae_loss":ae_loss, "kl_loss":kl_loss,
#         #         "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}

    def loss(self, input_data, fwd_out, true_energy):
        """
        - Overrides loss in gumboltCaloV5.py
        """
        logger.debug("loss")

        kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(fwd_out.post_logits, fwd_out.post_samples)
        # ae_loss = self._output_loss(input_data, fwd_out.output_activations) * torch.exp(self._config.model.mse_weight*input_data)
        sigma = 2 * torch.sqrt(torch.max(input_data, torch.min(input_data[input_data>0])))
        interpolation_param = self._config.model.interpolation_param
        ae_loss = torch.pow((input_data - fwd_out.output_activations)/sigma,2) * (1 - interpolation_param + interpolation_param*torch.pow(sigma,2)) * torch.exp(self._config.model.mse_weight*input_data)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0) * self._config.model.coefficient

        #hit_loss = self._hit_loss(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.))
        #hit_loss = torch.mean(torch.sum(hit_loss, dim=1), dim=0)
        # hit_loss = binary_cross_entropy_with_logits(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.), reduction='none')
        hit_loss = binary_cross_entropy_with_logits(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.), weight= (1+input_data).pow(self._config.model.bce_weights_power), reduction='none') #, weight= 1 + input_data: (1+input_data).sqrt()
        spIdx = torch.where(input_data > 0, 0., 1.).sum(dim=1) / input_data.shape[1]
        sparsity_weight = torch.exp(self._config.model.alpha - self._config.model.gamma * spIdx)
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1) * sparsity_weight, dim=0)

        # labels_target = nn.functional.one_hot(true_energy.divide(256).log2().to(torch.int64), num_classes=15).squeeze(1).to(torch.float)
        # hit_label = binary_cross_entropy_with_logits(fwd_out.labels, labels_target)


        # if self._config.engine.modelhits:
        return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
                "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}
        # else:
        #     return {"ae_loss":ae_loss, "kl_loss":kl_loss,
        #         "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}

    
    def kl_divergence(self, post_logits, post_samples, is_training=True):
        """Overrides kl_divergence in GumBolt.py

        :param post_logits (list) : List of f(logit_i|x, e) for each hierarchy
                                    layer i. Each element is a tensor of size
                                    (batch_size * n_nodes_per_hierarchy_layer)
        :param post_zetas (list) : List of q(zeta_i|x, e) for each hierarchy
                                   layer i. Each element is a tensor of size
                                   (batch_size * n_nodes_per_hierarchy_layer)
        """
        # Concatenate all hierarchy levels
        logits_q_z = torch.cat(post_logits, 1)
        post_zetas = torch.cat(post_samples, 1)

        # Compute cross-entropy b/w post_logits and post_samples
        # entropy = - self._bce_loss(logits_q_z, post_zetas)
        entropy = - self._bce_loss(logits_q_z, post_zetas[:,self._config.model.n_latent_nodes:])
        entropy = torch.mean(torch.sum(entropy, dim=1), dim=0)

        # Compute positive phase (energy expval under posterior variables) 
        n_nodes_p = self.prior.nodes_per_partition
        pos_energy = self.energy_exp_cond(post_zetas[:, :n_nodes_p],
                                     post_zetas[:, n_nodes_p:2*n_nodes_p],
                                     post_zetas[:, 2*n_nodes_p:3*n_nodes_p],
                                     post_zetas[:, 3*n_nodes_p:])

        # Compute gradient computation of the logZ term
        p0_state, p1_state, p2_state, p3_state \
            = self.sampler.block_gibbs_sampling_cond(post_zetas[:, :n_nodes_p],
                                     post_zetas[:, n_nodes_p:2*n_nodes_p],
                                     post_zetas[:, 2*n_nodes_p:3*n_nodes_p],
                                     post_zetas[:, 3*n_nodes_p:], method=self._config.model.rbmMethod)
        
        #beta, _, _, _ = self.find_beta()
        #beta = 7.5
        #p0_state, p1_state, p2_state, p3_state = self.dwave_sampling(num_samples=self._config.engine.rbm_batch_size, measure_time=False, beta=1.0/beta)

        
        # neg_energy = - self.energy_exp(p0_state, p1_state, p2_state, p3_state)
        neg_energy = - self.energy_exp_cond(p0_state, p1_state, p2_state, p3_state)

        # Estimate of the kl-divergence
        kl_loss = entropy + pos_energy + neg_energy
        return kl_loss, entropy, pos_energy, neg_energy

    
#     def energy_exp_cond(self, p0_state, p1_state, p2_state, p3_state):
#         """Energy expectation value under the 4-partite BM
#         Overrides energy_exp in gumbolt.py

#         :param p0_state (torch.Tensor) : (batch_size, n_nodes_p1)
#         :param p1_state (torch.Tensor) : (batch_size, n_nodes_p2)
#         :param p2_state (torch.Tensor) : (batch_size, n_nodes_p3)
#         :param p3_state (torch.Tensor) : (batch_size, n_nodes_p4)

#         :return energy expectation value over the current batch
#         """
#         w_dict = self.prior.weight_dict
#         b_dict = self.prior.bias_dict

#         w_dict_cp = {}

#         # Broadcast weight matrices (n_nodes_pa, n_nodes_pb) to
#         # (batch_size, n_nodes_pa, n_nodes_pb)
#         for key in w_dict.keys():
#             w_dict_cp[key] = w_dict[key] + torch.zeros((p0_state.size(0),) +
#                                                     w_dict[key].size(),
#                                                     device=w_dict[key].device)

#         # Prepare px_state_t for torch.bmm()
#         # Change px_state.size() to (batch_size, 1, n_nodes_px)
#         p0_state_t = p0_state.unsqueeze(2).permute(0, 2, 1)
#         p1_state_t = p1_state.unsqueeze(2).permute(0, 2, 1)
#         p2_state_t = p2_state.unsqueeze(2).permute(0, 2, 1)

#         # Prepare py_state for torch.bmm()
#         # Change py_state.size() to (batch_size, n_nodes_py, 1)
#         p1_state_i = p1_state.unsqueeze(2)
#         p2_state_i = p2_state.unsqueeze(2)
#         p3_state_i = p3_state.unsqueeze(2)

#         # Compute the energies for batch samples
#         batch_energy = -torch.matmul(p1_state, b_dict['1']) - \
#             torch.matmul(p2_state, b_dict['2']) - \
#             torch.matmul(p3_state, b_dict['3']) - \
#             torch.bmm(p0_state_t,
#                       torch.bmm(w_dict_cp['01'], p1_state_i)).reshape(-1) - \
#             torch.bmm(p0_state_t,
#                       torch.bmm(w_dict_cp['02'], p2_state_i)).reshape(-1) - \
#             torch.bmm(p0_state_t,
#                       torch.bmm(w_dict_cp['03'], p3_state_i)).reshape(-1) - \
#             torch.bmm(p1_state_t,
#                       torch.bmm(w_dict_cp['12'], p2_state_i)).reshape(-1) - \
#             torch.bmm(p1_state_t,
#                       torch.bmm(w_dict_cp['13'], p3_state_i)).reshape(-1) - \
#             torch.bmm(p2_state_t,
#                       torch.bmm(w_dict_cp['23'], p3_state_i)).reshape(-1)

#         return torch.mean(batch_energy, dim=0)
    
#     def dwave_sampling(self, num_samples=64, measure_time=False, beta=1.0):
#         """
        
#         Generate samples from DWave
#         """

#         # Extract the RBM parameters
#         prbm_weights = {}
#         prbm_bias = {}
#         for key in self.prior._weight_dict.keys():
#             prbm_weights[key] = self.prior._weight_dict[key]
#         for key in self.prior._bias_dict.keys():
#             prbm_bias[key] = self.prior._bias_dict[key]
#         prbm_edgelist = self.prior._pruned_edge_list


#         if self.prior.idx_dict is None:
#             idx_dict, device = self.prior.gen_qubit_idx_dict()
#         else:
#             idx_dict = self.prior.idx_dict

#         qubit_idxs = idx_dict['0'] + idx_dict['1'] + idx_dict['2'] + idx_dict['3']

#         idx_map = {}
#         for key in idx_dict.keys():
#             idx_map[key] = {idx:i for i, idx in enumerate(self.prior.idx_dict[key])}

#         dwave_weights = {}
#         dwave_bias = {}
#         for key in prbm_weights.keys():
#             dwave_weights[key] = - prbm_weights[key]/4. * beta
#         for key in prbm_bias.keys():
#             s = torch.zeros(prbm_bias[key].size(), device=prbm_bias[key].device)
#             for i in range(4):
#                 if i > int(key):
#                     wKey = key + str(i)
#                     s = s - torch.sum(prbm_weights[wKey], dim=1)/4. * beta
#                 elif i < int(key):
#                     wKey = str(i) + key
#                     s = s - torch.sum(prbm_weights[wKey], dim=0)/4. * beta
#             dwave_bias[key] = - prbm_bias[key]/2.0 * beta + s



#         dwave_weights_np = {}
#         for key in dwave_weights.keys():
#             dwave_weights_np[key] = dwave_weights[key].detach().cpu().numpy()
#         biases = torch.cat([dwave_bias[key] for key in dwave_bias.keys()])

#         # Initialize the values of biases and couplers. The next lines are critical
#         # maps the RBM coupling values into dwave's couplings h, J. In particular,
#         # J is a dictionary, each key is an edge in Pegasus
#         h = {qubit_idx:bias for qubit_idx, bias in zip(qubit_idxs, biases)}
#         J = {}
#         for edge in prbm_edgelist:
#             partition_edge_0 = self.find_partition_key(edge[0], idx_dict)
#             partition_edge_1 = self.find_partition_key(edge[1], idx_dict)
#             if int(partition_edge_0) < int(partition_edge_1):
#                 wKey = partition_edge_0 + partition_edge_1
#                 J[edge] = dwave_weights_np[wKey][idx_map[partition_edge_0][edge[0]]][idx_map[partition_edge_1][edge[1]]]
#             elif int(partition_edge_0) > int(partition_edge_1):
#                 wKey = partition_edge_1 + partition_edge_0
#                 J[edge] = dwave_weights_np[wKey][idx_map[partition_edge_1][edge[1]]][idx_map[partition_edge_0][edge[0]]]

#         if measure_time:
#             start = time.perf_counter()
#             response = self._qpu_sampler.sample_ising(h, J, num_reads=num_samples, answer_mode='raw', auto_scale=False)
#             self.sampling_time_qpu.append([time.perf_counter() - start, num_samples])
#         else:
#             response = self._qpu_sampler.sample_ising(h, J, num_reads=num_samples, answer_mode='raw', auto_scale=False)

#         dwave_samples, dwave_energies, origSamples = self.batch_dwave_samples(response, qubit_idxs)
#         # dwave_samples, dwave_energies = response.record['sample'], response.record['energy']
#         dwave_samples = torch.tensor(dwave_samples, dtype=torch.float).to(prbm_weights['01'].device)

#         # Convert spin Ising samples to binary RBM samples
#         _ZERO = torch.tensor(0., dtype=torch.float).to(prbm_weights['01'].device)
#         _MINUS_ONE = torch.tensor(-1., dtype=torch.float).to(prbm_weights['01'].device)

#         dwave_samples = torch.where(dwave_samples == _MINUS_ONE, _ZERO, dwave_samples)

#         partition_size = self._config.model.n_latent_nodes
#         return dwave_samples[:,:partition_size], dwave_samples[:,partition_size:2*partition_size], dwave_samples[:,2*partition_size:3*partition_size], dwave_samples[:,3*partition_size:4*partition_size]
    
    
#     def generate_samples_qpu_cond(self, num_samples=64, true_energy=None, measure_time=False, beta=1.0, thrsh=20):
#         """
#         generate_samples()
        
#         Overrides generate samples in gumboltCaloV5.py
#         """
#         true_energies = []
#         samples = []
#         # Extract the RBM parameters
#         prbm_weights = {}
#         prbm_bias = {}
#         for key in self.prior._weight_dict.keys():
#             prbm_weights[key] = self.prior._weight_dict[key]
#         for key in self.prior._bias_dict.keys():
#             prbm_bias[key] = self.prior._bias_dict[key]
#         prbm_edgelist = self.prior._pruned_edge_list
        
#         if self.prior.idx_dict is None:
#             idx_dict, device = self.prior.gen_qubit_idx_dict()
#         else:
#             idx_dict = self.prior.idx_dict
        
#         qubit_idxs = idx_dict['0'] + idx_dict['1'] + idx_dict['2'] + idx_dict['3']
        
#         idx_map = {}
#         for key in idx_dict.keys():
#             idx_map[key] = {idx:i for i, idx in enumerate(self.prior.idx_dict[key])}
        
#         dwave_weights = {}
#         dwave_bias = {}
#         for key in prbm_weights.keys():
#             dwave_weights[key] = - prbm_weights[key]/4. * beta
#         for key in prbm_bias.keys():
#             s = torch.zeros(prbm_bias[key].size(), device=prbm_bias[key].device)
#             for i in range(4):
#                 if i > int(key):
#                     wKey = key + str(i)
#                     s = s - torch.sum(prbm_weights[wKey], dim=1)/4. * beta
#                 elif i < int(key):
#                     wKey = str(i) + key
#                     s = s - torch.sum(prbm_weights[wKey], dim=0)/4. * beta
#             dwave_bias[key] = - prbm_bias[key]/2.0 * beta + s
        
#         dwave_weights_np = {}
#         for key in dwave_weights.keys():
#             dwave_weights_np[key] = dwave_weights[key].detach().cpu().numpy()
#         biases = torch.cat([dwave_bias[key] for key in dwave_bias.keys()])
        
#         # Initialize the values of biases and couplers. The next lines are critical
#         # maps the RBM coupling values into dwave's couplings h, J. In particular,
#         # J is a dictionary, each key is an edge in Pegasus
#         h = {qubit_idx:bias for qubit_idx, bias in zip(qubit_idxs, biases)}
#         J = {}
#         for edge in prbm_edgelist:
#             partition_edge_0 = self.find_partition_key(edge[0], idx_dict)
#             partition_edge_1 = self.find_partition_key(edge[1], idx_dict)
#             if int(partition_edge_0) < int(partition_edge_1):
#                 wKey = partition_edge_0 + partition_edge_1
#                 J[edge] = dwave_weights_np[wKey][idx_map[partition_edge_0][edge[0]]][idx_map[partition_edge_1][edge[1]]]
#             elif int(partition_edge_0) > int(partition_edge_1):
#                 wKey = partition_edge_1 + partition_edge_0
#                 J[edge] = dwave_weights_np[wKey][idx_map[partition_edge_1][edge[1]]][idx_map[partition_edge_0][edge[0]]]
        
        
#         # fb[60] = flux_biases.h_to_fluxbias(-20)  
#         # fb[61] = flux_biases.h_to_fluxbias(20)    # I was able to go up to 50, but ~20 should be enough
#         # response = self._qpu_sampler.sample_ising(h, J, num_reads=num_samples, answer_mode='raw', auto_scale=False, flux_drift_compensation=False, flux_biases=fb)
        
        
#         # if measure_time:
#         #     # start = time.process_time()
#         #     start = time.perf_counter()
#         #     response = self._qpu_sampler.sample_ising(h, J, num_reads=num_samples, answer_mode='raw', auto_scale=False)
#         #     self.sampling_time_qpu.append([time.perf_counter() - start, num_samples])
#         #     # self.sampling_time_qpu.append([time.process_time() - start, num_samples])
#         # else:
#         response_list = []
#         for x in true_energy:
#             fb = self.gen_fb(x, thrsh=thrsh)
#             response_list.append( self._qpu_sampler.sample_ising(h, J, num_reads=1, answer_mode='raw', auto_scale=False, flux_drift_compensation=False, flux_biases=fb))
            
#         response_array = np.concatenate([response_list[i].record["sample"] for i in range(len(response_list))])

#         dwave_samples, dwave_energies, origSamples = self.batch_dwave_samples_cond(response_array, qubit_idxs)
#         # dwave_samples, dwave_energies = response.record['sample'], response.record['energy']
#         dwave_samples = torch.tensor(dwave_samples, dtype=torch.float).to(prbm_weights['01'].device)
        
#         # Convert spin Ising samples to binary RBM samples
#         _ZERO = torch.tensor(0., dtype=torch.float).to(prbm_weights['01'].device)
#         _MINUS_ONE = torch.tensor(-1., dtype=torch.float).to(prbm_weights['01'].device)
        
#         dwave_samples = torch.where(dwave_samples == _MINUS_ONE, _ZERO, dwave_samples)
#         self.dwave_samples = dwave_samples
        
#         if true_energy is None:
#             true_e = torch.rand((num_samples, 1), device=prbm_weights['01'].device).detach() * 100.
#         else:
#             true_e = torch.ones((num_samples, 1), device=prbm_weights['01'].device).detach() * true_energy
#         # prior_samples = torch.cat([dwave_samples, true_e], dim=1)
#         prior_samples = torch.cat([dwave_samples], dim=1)
#         self.prior_samples = prior_samples
            
#         # output_hits, output_activations = self.decoder(prior_samples)
#         output_hits, output_activations = self.decoder(prior_samples, true_e)
#         beta = torch.tensor(self._config.model.beta_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
#         # if self._config.engine.modelhits:
#         sample = self._inference_energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, False)
#         # else:
#         #     sample = self._inference_energy_activation_fct(output_activations) * torch.ones(output_hits.size(), device=output_hits.device) 
#         # samples = self._energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, False) 
#         true_energies.append(true_e)
#         samples.append(sample) 
#         # return torch.cat(true_energies, dim=0).unsqueeze(dim=1), samples
#         return torch.cat(true_energies, dim=0), torch.cat(samples, dim=0)
    
#     def gen_fb(self, x, thrsh=30):
#         fb = [0]*self._qpu_sampler.properties['num_qubits']
#         bin_energy = self.encoder.binary_energy(x.unsqueeze(0))
#         fb_lists = ((bin_energy.to(dtype=int) * 2 - 1) * (-1) * thrsh).cpu().numpy()[0,:]
#         for i,idx in enumerate(self.prior.idx_dict['0']):
#             fb[idx] = h_to_fluxbias(fb_lists[i])
#         return fb
    
#     def find_partition_key(self, idx, qubit_idxs):
#         for key in qubit_idxs.keys():
#             if idx in qubit_idxs[key]:
#                 return key
    
    # def generate_samples_cond(self, num_samples: int = 128, true_energy=None, measure_time=False):
    #     """Generate data samples by decoding RBM samples

    #     :param num_samples (int): No. of data samples to generate in one shot
    #     :param true_energy (int): Default None, Incident energy of the particle

    #     :return true_energies (torch.Tensor): Incident energies of the particle
    #     for each sample (num_samples,)
    #     :return samples (torch.Tensor): Data samples, (num_samples, *)
    #     """
    #     n_iter = max(num_samples//self.sampler.batch_size, 1)
    #     true_es, samples = [], []
    #     u = self.encoder.binary_energy(true_energy).to(dtype=torch.float32)

    #     for _ in range(n_iter):
    #         if measure_time:
    #             # start = time.process_time()
    #             start = time.perf_counter()
    #             p0_state, p1_state, p2_state, p3_state = self.sampler.block_gibbs_sampling_cond(p0=u)
    #             torch.cuda.current_stream().synchronize()
    #             self.sampling_time_gpu.append([time.perf_counter() - start, self.sampler.batch_size])
    #             # self.sampling_time_gpu.append([time.process_time() - start, self.sampler.batch_size])
    #         else:
    #             p0_state, p1_state, p2_state, p3_state = self.sampler.block_gibbs_sampling_cond(p0=u)

    #         if true_energy is None:
    #             # true_e ~ U[1, 100]
    #             true_e = (torch.rand((p0_state.size(0), 1),
    #                                  device=p0_state.device) * 99.) + 1.
    #         else:
    #             # true_e = true_energy
    #             true_e = torch.ones((p0_state.size(0), 1),
    #                                 device=p0_state.device) * true_energy
    #         # prior_samples = torch.cat([p0_state, p1_state, p2_state, p3_state,
    #         #                            true_e], dim=1)
    #         prior_samples = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1)

    #         hits, activations = self.decoder(prior_samples, true_e, self.act_fct_slope, self.x_raw)
    #         beta = torch.tensor(self._config.model.beta_smoothing_fct,
    #                             dtype=torch.float, device=hits.device)
    #         sample = self._inference_energy_activation_fct(activations) \
    #             * self._hit_smoothing_dist_mod(hits, beta, False)

    #         true_es.append(true_e)
    #         samples.append(sample)

    #     return torch.cat(true_es, dim=0), torch.cat(samples, dim=0)

    # def generate_samples(self, num_samples: int = 128, true_energy=None, measure_time=False):
    #     """Generate data samples by decoding RBM samples

    #     :param num_samples (int): No. of data samples to generate in one shot
    #     :param true_energy (int): Default None, Incident energy of the particle

    #     :return true_energies (torch.Tensor): Incident energies of the particle
    #     for each sample (num_samples,)
    #     :return samples (torch.Tensor): Data samples, (num_samples, *)
    #     """
    #     n_iter = max(num_samples//self.sampler.batch_size, 1)
    #     true_es, samples = [], []

    #     for _ in range(n_iter):
    #         if measure_time:
    #             # start = time.process_time()
    #             start = time.perf_counter()
    #             p0_state, p1_state, p2_state, p3_state = self.sampler.block_gibbs_sampling()
    #             torch.cuda.current_stream().synchronize()
    #             self.sampling_time_gpu.append([time.perf_counter() - start, self.sampler.batch_size])
    #             # self.sampling_time_gpu.append([time.process_time() - start, self.sampler.batch_size])
    #         else:
    #             p0_state, p1_state, p2_state, p3_state = self.sampler.block_gibbs_sampling()

    #         if true_energy is None:
    #             # true_e ~ U[1, 100]
    #             true_e = (torch.rand((p0_state.size(0), 1),
    #                                  device=p0_state.device) * 99.) + 1.
    #         else:
    #             # true_e = true_energy
    #             true_e = torch.ones((p0_state.size(0), 1),
    #                                 device=p0_state.device) * true_energy
    #         # prior_samples = torch.cat([p0_state, p1_state, p2_state, p3_state,
    #         #                            true_e], dim=1)
    #         prior_samples = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1)

    #         hits, activations = self.decoder(prior_samples, true_e, self.act_fct_slope, self.x_raw)
    #         beta = torch.tensor(self._config.model.beta_smoothing_fct,
    #                             dtype=torch.float, device=hits.device)
    #         sample = self._inference_energy_activation_fct(activations) \
    #             * self._hit_smoothing_dist_mod(hits, beta, False)

    #         true_es.append(true_e)
    #         samples.append(sample)

    #     return torch.cat(true_es, dim=0), torch.cat(samples, dim=0)

#     def ising_energy(self, p0_state, p1_state, p2_state, p3_state, weight_dict, bias_dict):
#             """Energy expectation value under the 4-partite BM
#             Overrides energy_exp in gumbolt.py

#             :param p0_state (torch.Tensor) : (batch_size, n_nodes_p1)
#             :param p1_state (torch.Tensor) : (batch_size, n_nodes_p2)
#             :param p2_state (torch.Tensor) : (batch_size, n_nodes_p3)
#             :param p3_state (torch.Tensor) : (batch_size, n_nodes_p4)

#             :return energy expectation value over the current batch
#             """
#             w_dict = weight_dict
#             b_dict = bias_dict

#             w_dict_cp = {}

#             # Broadcast weight matrices (n_nodes_pa, n_nodes_pb) to
#             # (batch_size, n_nodes_pa, n_nodes_pb)
#             for key in w_dict.keys():
#                 w_dict_cp[key] = w_dict[key] + torch.zeros((p0_state.size(0),) +
#                                                         w_dict[key].size(),
#                                                         device=w_dict[key].device)

#             # Prepare px_state_t for torch.bmm()
#             # Change px_state.size() to (batch_size, 1, n_nodes_px)
#             p0_state_t = p0_state.unsqueeze(2).permute(0, 2, 1)
#             p1_state_t = p1_state.unsqueeze(2).permute(0, 2, 1)
#             p2_state_t = p2_state.unsqueeze(2).permute(0, 2, 1)

#             # Prepare py_state for torch.bmm()
#             # Change py_state.size() to (batch_size, n_nodes_py, 1)
#             p1_state_i = p1_state.unsqueeze(2)
#             p2_state_i = p2_state.unsqueeze(2)
#             p3_state_i = p3_state.unsqueeze(2)

#             # Compute the energies for batch samples
#             batch_energy = torch.matmul(p0_state, b_dict['0']) + \
#                 torch.matmul(p1_state, b_dict['1']) + \
#                 torch.matmul(p2_state, b_dict['2']) + \
#                 torch.matmul(p3_state, b_dict['3']) + \
#                 torch.bmm(p0_state_t,
#                           torch.bmm(w_dict_cp['01'], p1_state_i)).reshape(-1) + \
#                 torch.bmm(p0_state_t,
#                           torch.bmm(w_dict_cp['02'], p2_state_i)).reshape(-1) + \
#                 torch.bmm(p0_state_t,
#                           torch.bmm(w_dict_cp['03'], p3_state_i)).reshape(-1) + \
#                 torch.bmm(p1_state_t,
#                           torch.bmm(w_dict_cp['12'], p2_state_i)).reshape(-1) + \
#                 torch.bmm(p1_state_t,
#                           torch.bmm(w_dict_cp['13'], p3_state_i)).reshape(-1) + \
#                 torch.bmm(p2_state_t,
#                           torch.bmm(w_dict_cp['23'], p3_state_i)).reshape(-1)

#             return batch_energy
        
#     def ising_model(self, beta=1.0):
#         """
#         generate_samples()

#         Overrides generate samples in gumboltCaloV5.py
#         """
#         true_energies = []
#         samples = []
#         # Extract the RBM parameters
#         prbm_weights = {}
#         prbm_bias = {}
#         for key in self.prior._weight_dict.keys():
#             prbm_weights[key] = self.prior._weight_dict[key]
#         for key in self.prior._bias_dict.keys():
#             prbm_bias[key] = self.prior._bias_dict[key]
#         prbm_edgelist = self.prior._pruned_edge_list

#         # crbm_weights = self.prior.weights
#         # crbm_vbias = self.prior.visible_bias
#         # crbm_hbias = self.prior.hidden_bias
#         # crbm_edgelist = self.prior.pruned_edge_list
#         if self.prior.idx_dict is None:
#             idx_dict, device = self.prior.gen_qubit_idx_dict()
#         else:
#             idx_dict = self.prior.idx_dict

#         qubit_idxs = idx_dict['0'] + idx_dict['1'] + idx_dict['2'] + idx_dict['3']

#         # qubit_idxs = self.prior.visible_qubit_idxs + self.prior.hidden_qubit_idxs

#         idx_map = {}
#         for key in idx_dict.keys():
#             idx_map[key] = {idx:i for i, idx in enumerate(self.prior.idx_dict[key])}

#         # visible_idx_map = {visible_qubit_idx:i for i, visible_qubit_idx in enumerate(self.prior.visible_qubit_idxs)}
#         # hidden_idx_map = {hidden_qubit_idx:i for i, hidden_qubit_idx in enumerate(self.prior.hidden_qubit_idxs)}

#         dwave_weights = {}
#         dwave_bias = {}
#         for key in prbm_weights.keys():
#             dwave_weights[key] = - prbm_weights[key]/4. * beta
#         for key in prbm_bias.keys():
#             s = torch.zeros(prbm_bias[key].size(), device=prbm_bias[key].device)
#             for i in range(4):
#                 if i > int(key):
#                     wKey = key + str(i)
#                     s = s - torch.sum(prbm_weights[wKey], dim=1)/4. * beta
#                 elif i < int(key):
#                     wKey = str(i) + key
#                     s = s - torch.sum(prbm_weights[wKey], dim=0)/4. * beta
#             dwave_bias[key] = - prbm_bias[key]/2.0 * beta + s

#         # for key in dwave_bias.keys():
#         #     dwave_bias[key] = torch.clamp(dwave_bias[key], min=-5., max=5.)
#         # for key in dwave_weights.keys():
#         #     dwave_weights[key] = torch.clamp(dwave_weights[key], min=-5., max=5.)


#         # dwave_weights = torch.clamp(dwave_weights, min=-2., max=1.)
#         # dwave_vbias = torch.clamp(dwave_vbias, min=-2., max=2.)
#         # dwave_hbias = torch.clamp(dwave_hbias, min=-2., max=2.)

#         dwave_weights_np = {}
#         for key in dwave_weights.keys():
#             dwave_weights_np[key] = dwave_weights[key].detach().cpu().numpy()
#         biases = torch.cat([dwave_bias[key] for key in dwave_bias.keys()])
#         # dwave_weights_np = dwave_weights.detach().cpu().numpy()
#         # biases = torch.cat((dwave_vbias, dwave_hbias)).detach().cpu().numpy()

#         # Initialize the values of biases and couplers. The next lines are critical
#         # maps the RBM coupling values into dwave's couplings h, J. In particular,
#         # J is a dictionary, each key is an edge in Pegasus
#         h = {qubit_idx:bias for qubit_idx, bias in zip(qubit_idxs, biases)}
#         J = {}
#         for edge in prbm_edgelist:
#             partition_edge_0 = self.find_partition_key(edge[0], idx_dict)
#             partition_edge_1 = self.find_partition_key(edge[1], idx_dict)
#             if int(partition_edge_0) < int(partition_edge_1):
#                 wKey = partition_edge_0 + partition_edge_1
#                 J[edge] = dwave_weights_np[wKey][idx_map[partition_edge_0][edge[0]]][idx_map[partition_edge_1][edge[1]]]
#             elif int(partition_edge_0) > int(partition_edge_1):
#                 wKey = partition_edge_1 + partition_edge_0
#                 J[edge] = dwave_weights_np[wKey][idx_map[partition_edge_1][edge[1]]][idx_map[partition_edge_0][edge[0]]]
#         return h, J, qubit_idxs, idx_dict, dwave_weights, dwave_bias
        
#     def find_beta(self, num_reads=128, beta_init=10.0, lr=0.01, num_epochs = 20, delta = 2.0, method = 1, TOL=True, const = 1.0, adaptive = True):
#         delta_init = delta
#         lr_init = lr
        
#         beta = beta_init
#         beta_list = []
#         rbm_energy_list = []
#         dwave_energies_list = []
#         mean_rbm_energy_list = []
#         mean_dwave_energy_list = []
#         training_results = {}
#         # sample_size = self.sampler._batch_size
#         # self.sampler._batch_size = num_reads * num_epochs
#         thrsh_met = 0

#         for epoch in range(num_epochs+1):
#             _,_,_,_, dwave_weights_rbm, dwave_bias_rbm = self.ising_model(1.0)
#             h, J, qubit_idxs, idx_dict, _, _ = self.ising_model(1.0 / beta)
#             if epoch == 0:
#                 p0_state, p1_state, p2_state, p3_state = self.sampler.block_gibbs_sampling()
#                 p0_ising = p0_state * 2 - 1
#                 p1_ising = p1_state * 2 - 1
#                 p2_ising = p2_state * 2 - 1
#                 p3_ising = p3_state * 2 - 1
#                 rbm_energies = self.ising_energy(p0_ising, p1_ising, p2_ising, p3_ising, dwave_weights_rbm, dwave_bias_rbm)
#                 rbm_energies = rbm_energies.detach().cpu().numpy()


#             response = self._qpu_sampler.sample_ising(h, J, num_reads=num_reads, auto_scale=False)
#             dwave_samples, dwave_energies, origSamples = self.batch_dwave_samples(response, qubit_idxs)
#             # dwave_samples, dwave_energies = response.record['sample'], response.record['energy']

#             nonpl = len(idx_dict['0'])
#             dwave_1, dwave_2, dwave_3, dwave_4 = dwave_samples[:,0:nonpl], dwave_samples[:,nonpl:2*nonpl], dwave_samples[:,2*nonpl:3*nonpl], dwave_samples[:,3*nonpl:4*nonpl]
#             dwave_1_t = torch.tensor(dwave_1).to(p0_ising.device).float()
#             dwave_2_t = torch.tensor(dwave_2).to(p0_ising.device).float()
#             dwave_3_t = torch.tensor(dwave_3).to(p0_ising.device).float()
#             dwave_4_t = torch.tensor(dwave_4).to(p0_ising.device).float()
#             dwave_energies = self.ising_energy(dwave_1_t, dwave_2_t, dwave_3_t, dwave_4_t, dwave_weights_rbm, dwave_bias_rbm)
#             dwave_energies = dwave_energies.detach().cpu().numpy()
#             mean_rbm_energy = np.mean(rbm_energies)
#             mean_dwave_energy = np.mean(dwave_energies)

#             rbm_energy_list.append(rbm_energies)
#             dwave_energies_list.append(dwave_energies)
#             mean_rbm_energy_list.append(mean_rbm_energy)
#             mean_dwave_energy_list.append(mean_dwave_energy)
#             beta_list.append(beta)
#             # print (f'Epoch {epoch}: beta = {beta}')
#             logger.info(f'Epoch {epoch}: beta = {beta}')
#             if method == 1:
#                 if adaptive:
#                     lr = np.max([lr_init, np.power(beta,2)/np.var(dwave_energies)])
#                 beta = beta - lr * (mean_dwave_energy - mean_rbm_energy)
#             else:
#                 if adaptive:
#                     delta = np.max([delta_init, np.abs(mean_dwave_energy)/np.var(dwave_energies)])
#                 beta = np.max([2.0, beta * np.power(mean_dwave_energy/mean_rbm_energy, delta)])
            
#             if TOL and np.abs(mean_rbm_energy - mean_dwave_energy) < const * 2.0 * np.std(dwave_energies) * np.std(rbm_energies) / ( np.sqrt(num_reads) * (np.std(dwave_energies) + np.std(rbm_energies))):
#                 thrsh_met = 1
#                 break
#         beta = beta_list[-1]
#         # self.sampler._batch_size = sample_size
#         return beta, beta_list, rbm_energy_list, dwave_energies_list, thrsh_met
    
#     def remove_gaps(self, lst):
#         # Create a sorted set of the unique elements in the list
#         unique_sorted = sorted(set(lst))

#         # Create a dictionary that maps each element to its index in the sorted set
#         rank_dict = {val: idx for idx, val in enumerate(unique_sorted)}

#         # Replace each element in the original list with its rank
#         ranked_lst = [rank_dict[val] for val in lst]

#         return ranked_lst

#     def create_sparse_matrix(self, lst):
#         # Example 1D array of column positions
#         column_positions = torch.tensor(lst)

#         # Number of rows
#         num_rows = len(column_positions)

#         # Creating row indices since each 1 is in a different row
#         row_indices = torch.arange(num_rows)

#         # Coordinates for non-zero values
#         indices = torch.stack([row_indices, column_positions])

#         # Creating the data array, all values are 1
#         values = torch.ones(num_rows)

#         # Create the sparse matrix
#         sparse_matrix = torch.sparse_coo_tensor(indices, values, (num_rows, num_rows))

#         # Convert to dense for visualization
#         dense_matrix = sparse_matrix.to_dense()
#         return dense_matrix
    
#     def batch_dwave_samples_cond(self, response, qubit_idxs):
#         """
#         This replaces gumboltAtlasCRBMCNN.
#         After Hao figured we could use response.record instead of response.data.
#         """
#         original_list = qubit_idxs
#         sequential_qubit_idxs = self.remove_gaps(qubit_idxs)
#         dense_matrix = self.create_sparse_matrix(sequential_qubit_idxs)
        
#         batch_samples = torch.mm(dense_matrix, torch.tensor(response).transpose(0,1).float()).transpose(0,1)

#         return batch_samples.numpy(), 0, 0