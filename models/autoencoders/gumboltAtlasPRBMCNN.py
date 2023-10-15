"""
GumBolt implementation for Calorimeter data
CNN - Changed to CNN encoder creation
"""
# Torch imports
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn as nn 

from models.samplers.GibbsSampling import GS

# DiVAE.models imports
from models.autoencoders.gumboltAtlasCRBMCNNDecCond import GumBoltAtlasCRBMCNNDCond
from CaloQVAE.models.rbm import pegasusRBM
from CaloQVAE.models.samplers import pgbs
# from models.networks.EncoderCNN import EncoderCNN
from models.networks.EncoderUCNN import EncoderUCNN
from models.networks.basicCoders import DecoderCNNCond

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class GumBoltAtlasPRBMCNN(GumBoltAtlasCRBMCNNDCond):
    """
    GumBolt
    """

    def __init__(self, **kwargs):
        super(GumBoltAtlasPRBMCNN, self).__init__(**kwargs)
        self._model_type = "GumBoltAtlasPRBMCNN"
        self._bce_loss = BCEWithLogitsLoss(reduction="none")
        
    def _create_prior(self):
        """Override _create_prior in GumBoltCaloV6.py

        :return: Instance of a PegasusRBM
        """
        assert (self._config.model.n_latent_hierarchy_lvls *
                self._config.model.n_latent_nodes) % 4 == 0, \
            'total no. of latent nodes should be divisible by 4'

        nodes_per_partition = int((self._config.model.n_latent_hierarchy_lvls *
                                   self._config.model.n_latent_nodes)/4)
        
        return pegasusRBM.PegasusRBM(nodes_per_partition)
        
    def _create_sampler(self, rbm=None):
        """Override _create_sampler in GumBoltCaloV6.py

        :return: Instance of a PGBS sampler
        """
        return pgbs.PGBS(self.prior, self._config.engine.rbm_batch_size,
                         n_steps=self._config.engine.n_gibbs_sampling_steps)

    
#     def forward(self, xx, is_training, beta_smoothing_fct=5):
#         """
#         - Overrides forward in GumBoltAtlasCRBMCNN.py
        
#         Returns:
#             out: output container 
#         """
#         logger.debug("forward")
        
#         #see definition for explanation
#         out=self._output_container.clear()
#         x, x0 = xx
        
# 	    #Step 1: Feed data through encoder
#         # in_data = torch.cat([x[0], x[1]], dim=1)
        
#         out.beta, out.post_logits, out.post_samples = self.encoder(x, x0, is_training, beta_smoothing_fct)
#         # out.post_samples = self.encoder(x, x0, is_training)
#         post_samples = out.post_samples
#         post_samples = torch.cat(out.post_samples, 1)
# #         post_samples = torch.cat([post_samples, x[1]], dim=1)
        
#         output_hits, output_activations = self.decoder(post_samples, x0)
#         # labels = self.classifier(output_hits)
        
#         out.output_hits = output_hits
#         # out.labels = labels
#         beta = torch.tensor(self._config.model.output_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
#         out.output_activations = self._energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
#         return out
    
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
        entropy = - self._bce_loss(logits_q_z, post_zetas)
        entropy = torch.mean(torch.sum(entropy, dim=1), dim=0)

        # Compute positive phase (energy expval under posterior variables) 
        n_nodes_p = self.prior.nodes_per_partition
        pos_energy = self.energy_exp(post_zetas[:, :n_nodes_p],
                                     post_zetas[:, n_nodes_p:2*n_nodes_p],
                                     post_zetas[:, 2*n_nodes_p:3*n_nodes_p],
                                     post_zetas[:, 3*n_nodes_p:])

        # Compute gradient computation of the logZ term
        p0_state, p1_state, p2_state, p3_state \
            = self.sampler.block_gibbs_sampling()
        neg_energy = - self.energy_exp(p0_state, p1_state, p2_state, p3_state)

        # Estimate of the kl-divergence
        kl_loss = entropy + pos_energy + neg_energy
        return kl_loss, entropy, pos_energy, neg_energy

    def energy_exp(self, p0_state, p1_state, p2_state, p3_state):
        """Energy expectation value under the 4-partite BM
        Overrides energy_exp in gumbolt.py

        :param p0_state (torch.Tensor) : (batch_size, n_nodes_p1)
        :param p1_state (torch.Tensor) : (batch_size, n_nodes_p2)
        :param p2_state (torch.Tensor) : (batch_size, n_nodes_p3)
        :param p3_state (torch.Tensor) : (batch_size, n_nodes_p4)

        :return energy expectation value over the current batch
        """
        w_dict = self.prior._weight_dict
        b_dict = self.prior._bias_dict

        w_dict_cp = {}

        # Broadcast weight matrices (n_nodes_pa, n_nodes_pb) to
        # (batch_size, n_nodes_pa, n_nodes_pb)
        for key in w_dict.keys():
            w_dict_cp[key] = w_dict[key] + torch.zeros((p0_state.size(0),) +
                                                    w_dict[key].size(),
                                                    device=w_dict[key].device)

        # Prepare px_state_t for torch.bmm()
        # Change px_state.size() to (batch_size, 1, n_nodes_px)
        p0_state_t = p0_state.unsqueeze(2).permute(0, 2, 1)
        p1_state_t = p1_state.unsqueeze(2).permute(0, 2, 1)
        p2_state_t = p2_state.unsqueeze(2).permute(0, 2, 1)

        # Prepare py_state for torch.bmm()
        # Change py_state.size() to (batch_size, n_nodes_py, 1)
        p1_state_i = p1_state.unsqueeze(2)
        p2_state_i = p2_state.unsqueeze(2)
        p3_state_i = p3_state.unsqueeze(2)

        # Compute the energies for batch samples
        batch_energy = -torch.matmul(p0_state, b_dict['0']) - \
            torch.matmul(p1_state, b_dict['1']) - \
            torch.matmul(p2_state, b_dict['2']) - \
            torch.matmul(p3_state, b_dict['3']) - \
            torch.bmm(p0_state_t,
                      torch.bmm(w_dict_cp['01'], p1_state_i)).reshape(-1) - \
            torch.bmm(p0_state_t,
                      torch.bmm(w_dict_cp['02'], p2_state_i)).reshape(-1) - \
            torch.bmm(p0_state_t,
                      torch.bmm(w_dict_cp['03'], p3_state_i)).reshape(-1) - \
            torch.bmm(p1_state_t,
                      torch.bmm(w_dict_cp['12'], p2_state_i)).reshape(-1) - \
            torch.bmm(p1_state_t,
                      torch.bmm(w_dict_cp['13'], p3_state_i)).reshape(-1) - \
            torch.bmm(p2_state_t,
                      torch.bmm(w_dict_cp['23'], p3_state_i)).reshape(-1)

        return torch.mean(batch_energy, dim=0)
    
    def generate_samples_qpu(self, num_samples=64, true_energy=None):
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
            dwave_weights[key] = - prbm_weights[key]/4.
        for key in prbm_bias.keys():
            s = torch.zeros(prbm_bias[key].size(), device=prbm_bias[key].device)
            for i in range(4):
                if i > int(key):
                    wKey = key + str(i)
                    s = s - torch.sum(prbm_weights[wKey], dim=1)/4.
                elif i < int(key):
                    wKey = str(i) + key
                    s = s - torch.sum(prbm_weights[wKey], dim=0)/4.
            dwave_bias[key] = - prbm_bias[key]/2.0 + s
            
        for key in dwave_bias.keys():
            dwave_bias[key] = torch.clamp(dwave_bias[key], min=-2., max=2.)
        for key in dwave_weights.keys():
            dwave_weights[key] = torch.clamp(dwave_weights[key], min=-2., max=1.)
        # Convert the RBM parameters into Ising parameters
        # dwave_weights = -(crbm_weights/4.)
        # dwave_vbias = -(crbm_vbias/2. + torch.sum(crbm_weights, dim=1)/4.)
        # dwave_hbias = -(crbm_hbias/2. + torch.sum(crbm_weights, dim=0)/4.)
        
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
        # prior_samples = torch.cat([dwave_samples, true_e], dim=1)
        prior_samples = torch.cat([dwave_samples], dim=1)
        self.prior_samples = prior_samples
            
        # output_hits, output_activations = self.decoder(prior_samples)
        output_hits, output_activations = self.decoder(prior_samples, true_e)
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
    
    def generate_samples(self, num_samples: int = 128, true_energy=None):
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

            hits, activations = self.decoder(prior_samples, true_e)
            beta = torch.tensor(self._config.model.beta_smoothing_fct,
                                dtype=torch.float, device=hits.device)
            sample = self._energy_activation_fct(activations) \
                * self._hit_smoothing_dist_mod(hits, beta, False)

            true_es.append(true_e)
            samples.append(sample)

        return torch.cat(true_es, dim=0), torch.cat(samples, dim=0)

    def loss(self, input_data, fwd_out, true_energy):
        """
        - Overrides loss in gumboltCaloV5.py
        """
        logger.debug("loss")
        
        kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(fwd_out.post_logits, fwd_out.post_samples)
        # ae_loss = self._output_loss(input_data, fwd_out.output_activations) * torch.exp(self._config.model.mse_weight*input_data)
        sigma = torch.max(torch.sqrt(input_data), torch.tensor([0.1], device=input_data.device))
        interpolation_param = self._config.model.interpolation_param
        ae_loss = torch.pow((input_data - fwd_out.output_activations)/sigma,2) * (1 - interpolation_param + interpolation_param*torch.pow(sigma,2)) * torch.exp(self._config.model.mse_weight*input_data)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0) # <---- divide by sqrt(x)
        # torch.min(x[x>0])
        
        hit_loss = binary_cross_entropy_with_logits(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.), reduction='none')
        spIdx = torch.where(input_data > 0, 0., 1.).sum(dim=1) / input_data.shape[1]
        sparsity_weight = torch.exp(self._config.model.alpha - self._config.model.gamma * spIdx)
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1) * sparsity_weight, dim=0)

        
        # return {"ae_loss":ae_loss, "kl_loss":kl_loss,
        #         "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}
        
        return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
                "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}
        
        # return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
        #         "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy, "label_loss":hit_label}



