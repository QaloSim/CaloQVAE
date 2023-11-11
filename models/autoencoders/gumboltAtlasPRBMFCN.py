"""
GumBolt implementation for Calorimeter data
CNN - Changed to CNN encoder creation
"""
# Torch imports
import torch

# DiVAE.models imports
from models.autoencoders.gumboltAtlasPRBMCNN import GumBoltAtlasPRBMCNN
from models.networks.basicCoders import BasicDecoderV3
from models.networks.hierarchicalEncoderV2 import HierarchicalEncoderV2

import time

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class GumBoltAtlasPRBMFCN(GumBoltAtlasPRBMCNN):
    """
    GumBolt
    """

    def __init__(self, **kwargs):
        super(GumBoltAtlasPRBMCNN, self).__init__(**kwargs)
        self._model_type = "GumBoltAtlasPRBMFCN"
        # self._bce_loss = BCEWithLogitsLoss(reduction="none")
        self.sampling_time_qpu = []
        self.sampling_time_gpu = []
        
    def _create_decoder(self):
        logger.debug("GumBoltAtlasPRBMFCN::_create_decoder")
        self._decoder_nodes[0] = (self._decoder_nodes[0][0]+1,
                                  self._decoder_nodes[0][1])
        return BasicDecoderV3(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct,
                              cfg=self._config)
    
    def _create_encoder(self):
        """
        - Overrides _create_encoder in GumBoltCaloV5.py

        Returns:
            Hierarchical Encoder instance
        """
        logger.debug("GumBoltCaloV6::_create_encoder")
        return HierarchicalEncoderV2(
            input_dimension=self._flat_input_size+1,
            n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
            n_latent_nodes=self.n_latent_nodes,
            skip_latent_layer=False,
            smoother="Gumbel",
            cfg=self._config)
    
    def forward(self, xx, is_training, beta_smoothing_fct=5, act_fct_slope=0.02):
        """
        - Overrides forward in GumBoltCaloV5.py
        
        Returns:
            out: output container 
        """
        logger.debug("forward")
        
        #see definition for explanation
        out=self._output_container.clear()
        # x, x0 = xx
        
	    #Step 1: Feed data through encoder
        in_data = torch.cat(xx, dim=1)
        
        out.beta, out.post_logits, out.post_samples = self.encoder(in_data, is_training, beta_smoothing_fct)
        # out.post_samples = self.encoder(x, x0, is_training)
        post_samples = out.post_samples
        post_samples = torch.cat(out.post_samples, 1)
        post_samples = torch.cat([post_samples, xx[1]], dim=1)
        
        output_hits, output_activations = self.decoder(post_samples)
        # labels = self.classifier(output_hits)
        
        out.output_hits = output_hits

        beta = torch.tensor(self._config.model.output_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
        if self._config.engine.modelhits:
            if is_training:
                # out.output_activations = self._energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
                activation_fct_annealed = self._training_activation_fct(act_fct_slope)
                out.output_activations = activation_fct_annealed(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
            else:
                out.output_activations = self._inference_energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
        else:
            if is_training:
                # out.output_activations = self._energy_activation_fct(output_activations) * torch.ones(output_hits.size(), device=output_hits.device)
                activation_fct_annealed = self._training_activation_fct(act_fct_slope)
                out.output_activations = activation_fct_annealed(output_activations) * torch.ones(output_hits.size(), device=output_hits.device)
            else:
                out.output_activations = self._inference_energy_activation_fct(output_activations) *torch.ones(output_hits.size(), device=output_hits.device)
            # out.output_activations = self._energy_activation_fct(output_activations) * torch.ones(output_hits.size(), device=output_hits.device)
        return out
        
    
    def generate_samples_qpu(self, num_samples=64, true_energy=None, measure_time=False, beta=1.0):
        """
        generate_samples()
        
        Overrides generate samples in gumboltCaloV5.py
        """
        true_energies = []
        samples = []
        # Extract the RBM parameters
        prbm_weights = {}
        prbm_bias = {}
        for key in self.prior._weight_dict.keys():
            prbm_weights[key] = self.prior._weight_dict[key]
        for key in self.prior._bias_dict.keys():
            prbm_bias[key] = self.prior._bias_dict[key]
        prbm_edgelist = self.prior._pruned_edge_list
        
        # crbm_weights = self.prior.weights
        # crbm_vbias = self.prior.visible_bias
        # crbm_hbias = self.prior.hidden_bias
        # crbm_edgelist = self.prior.pruned_edge_list
        if self.prior.idx_dict is None:
            idx_dict, device = self.prior.gen_qubit_idx_dict()
        else:
            idx_dict = self.prior.idx_dict
        
        qubit_idxs = idx_dict['0'] + idx_dict['1'] + idx_dict['2'] + idx_dict['3']
        
        # qubit_idxs = self.prior.visible_qubit_idxs + self.prior.hidden_qubit_idxs
        
        idx_map = {}
        for key in idx_dict.keys():
            idx_map[key] = {idx:i for i, idx in enumerate(self.prior.idx_dict[key])}
        
        # visible_idx_map = {visible_qubit_idx:i for i, visible_qubit_idx in enumerate(self.prior.visible_qubit_idxs)}
        # hidden_idx_map = {hidden_qubit_idx:i for i, hidden_qubit_idx in enumerate(self.prior.hidden_qubit_idxs)}
        
        dwave_weights = {}
        dwave_bias = {}
        for key in prbm_weights.keys():
            dwave_weights[key] = - prbm_weights[key]/4. * beta
        for key in prbm_bias.keys():
            s = torch.zeros(prbm_bias[key].size(), device=prbm_bias[key].device)
            for i in range(4):
                if i > int(key):
                    wKey = key + str(i)
                    s = s - torch.sum(prbm_weights[wKey], dim=1)/4. * beta
                elif i < int(key):
                    wKey = str(i) + key
                    s = s - torch.sum(prbm_weights[wKey], dim=0)/4. * beta
            dwave_bias[key] = - prbm_bias[key]/2.0 * beta + s
            
        # for key in dwave_bias.keys():
        #     dwave_bias[key] = torch.clamp(dwave_bias[key], min=-5., max=5.)
        # for key in dwave_weights.keys():
        #     dwave_weights[key] = torch.clamp(dwave_weights[key], min=-5., max=5.)
        
        
        # dwave_weights = torch.clamp(dwave_weights, min=-2., max=1.)
        # dwave_vbias = torch.clamp(dwave_vbias, min=-2., max=2.)
        # dwave_hbias = torch.clamp(dwave_hbias, min=-2., max=2.)
        
        dwave_weights_np = {}
        for key in dwave_weights.keys():
            dwave_weights_np[key] = dwave_weights[key].detach().cpu().numpy()
        biases = torch.cat([dwave_bias[key] for key in dwave_bias.keys()])
        # dwave_weights_np = dwave_weights.detach().cpu().numpy()
        # biases = torch.cat((dwave_vbias, dwave_hbias)).detach().cpu().numpy()
        
        # Initialize the values of biases and couplers. The next lines are critical
        # maps the RBM coupling values into dwave's couplings h, J. In particular,
        # J is a dictionary, each key is an edge in Pegasus
        h = {qubit_idx:bias for qubit_idx, bias in zip(qubit_idxs, biases)}
        J = {}
        for edge in prbm_edgelist:
            partition_edge_0 = self.find_partition_key(edge[0], idx_dict)
            partition_edge_1 = self.find_partition_key(edge[1], idx_dict)
            if int(partition_edge_0) < int(partition_edge_1):
                wKey = partition_edge_0 + partition_edge_1
                J[edge] = dwave_weights_np[wKey][idx_map[partition_edge_0][edge[0]]][idx_map[partition_edge_1][edge[1]]]
            elif int(partition_edge_0) > int(partition_edge_1):
                wKey = partition_edge_1 + partition_edge_0
                J[edge] = dwave_weights_np[wKey][idx_map[partition_edge_1][edge[1]]][idx_map[partition_edge_0][edge[0]]]
        
        
        # for edge in crbm_edgelist:
        #     if edge[0] in self.prior.visible_qubit_idxs:
        #         J[edge] = dwave_weights_np[visible_idx_map[edge[0]]][hidden_idx_map[edge[1]]]
        #     else:
        #         J[edge] = dwave_weights_np[visible_idx_map[edge[1]]][hidden_idx_map[edge[0]]]
        
        
        if measure_time:
            # start = time.process_time()
            start = time.perf_counter()
            response = self._qpu_sampler.sample_ising(h, J, num_reads=num_samples, auto_scale=False)
            self.sampling_time_qpu.append([time.perf_counter() - start, num_samples])
            # self.sampling_time_qpu.append([time.process_time() - start, num_samples])
        else:
            response = self._qpu_sampler.sample_ising(h, J, num_reads=num_samples, auto_scale=False)

        dwave_samples, dwave_energies, origSamples = self.batch_dwave_samples(response, qubit_idxs)
        dwave_samples = torch.tensor(dwave_samples, dtype=torch.float).to(prbm_weights['01'].device)
        
        # Convert spin Ising samples to binary RBM samples
        _ZERO = torch.tensor(0., dtype=torch.float).to(prbm_weights['01'].device)
        _MINUS_ONE = torch.tensor(-1., dtype=torch.float).to(prbm_weights['01'].device)
        
        dwave_samples = torch.where(dwave_samples == _MINUS_ONE, _ZERO, dwave_samples)
        self.dwave_samples = dwave_samples
        
        if true_energy is None:
            true_e = torch.rand((num_samples, 1), device=prbm_weights['01'].device).detach() * 100.
        else:
            true_e = torch.ones((num_samples, 1), device=prbm_weights['01'].device).detach() * true_energy
        prior_samples = torch.cat([dwave_samples, true_e], dim=1)
        # prior_samples = torch.cat([dwave_samples], dim=1)
        self.prior_samples = prior_samples
            
        output_hits, output_activations = self.decoder(prior_samples)
        # output_hits, output_activations = self.decoder(prior_samples, true_e)
        beta = torch.tensor(self._config.model.beta_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
        if self._config.engine.modelhits:
            sample = self._inference_energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, False)
        else:
            sample = self._inference_energy_activation_fct(output_activations) * torch.ones(output_hits.size(), device=output_hits.device) 
        # samples = self._energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, False) 
        true_energies.append(true_e)
        samples.append(sample) 
        # return torch.cat(true_energies, dim=0).unsqueeze(dim=1), samples
        return torch.cat(true_energies, dim=0), torch.cat(samples, dim=0)
    
    def find_partition_key(self, idx, qubit_idxs):
        for key in qubit_idxs.keys():
            if idx in qubit_idxs[key]:
                return key
    
    def generate_samples(self, num_samples: int = 128, true_energy=None, measure_time=False):
        """Generate data samples by decoding RBM samples

        :param num_samples (int): No. of data samples to generate in one shot
        :param true_energy (int): Default None, Incident energy of the particle

        :return true_energies (torch.Tensor): Incident energies of the particle
        for each sample (num_samples,)
        :return samples (torch.Tensor): Data samples, (num_samples, *)
        """
        n_iter = max(num_samples//self.sampler.batch_size, 1)
        true_es, samples = [], []

        for _ in range(n_iter):
            if measure_time:
                # start = time.process_time()
                start = time.perf_counter()
                p0_state, p1_state, p2_state, p3_state = self.sampler.block_gibbs_sampling()
                torch.cuda.current_stream().synchronize()
                self.sampling_time_gpu.append([time.perf_counter() - start, self.sampler.batch_size])
                # self.sampling_time_gpu.append([time.process_time() - start, self.sampler.batch_size])
            else:
                p0_state, p1_state, p2_state, p3_state = self.sampler.block_gibbs_sampling()

            if true_energy is None:
                # true_e ~ U[1, 100]
                true_e = (torch.rand((p0_state.size(0), 1),
                                     device=p0_state.device) * 99.) + 1.
            else:
                # true_e = true_energy
                true_e = torch.ones((p0_state.size(0), 1),
                                    device=p0_state.device) * true_energy
            # prior_samples = torch.cat([p0_state, p1_state, p2_state, p3_state,
            #                            true_e], dim=1)
            prior_samples = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1)
            
            post_samples = torch.cat([prior_samples,true_e], dim=1)
        
            hits, activations = self.decoder(post_samples)
            beta = torch.tensor(self._config.model.beta_smoothing_fct,
                                dtype=torch.float, device=hits.device)
            sample = self._inference_energy_activation_fct(activations) \
                * self._hit_smoothing_dist_mod(hits, beta, False)

            true_es.append(true_e)
            samples.append(sample)

        return torch.cat(true_es, dim=0), torch.cat(samples, dim=0)

