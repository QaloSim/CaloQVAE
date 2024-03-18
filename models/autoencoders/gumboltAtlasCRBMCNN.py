"""
GumBolt implementation for Calorimeter data
CNN - Changed to CNN encoder creation
"""
# Torch imports
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn import LeakyReLU, ReLU
import torch.nn as nn 
import numpy as np

from models.samplers.GibbsSampling import GS

# DiVAE.models imports
# from models.rbm.qimeraRBM import QimeraRBM
from models.rbm.chimerav2 import QimeraRBM
from models.autoencoders.gumboltCaloCRBM import GumBoltCaloCRBM
# from models.networks.EncoderCNN import EncoderCNN
from models.networks.EncoderUCNN import EncoderUCNN, EncoderUCNNH, EncoderUCNNHPosEnc
from models.networks.basicCoders import DecoderCNN, Classifier, DecoderCNNCond, DecoderCNNCondSmall, DecoderCNNUnconditioned, DecoderCNNPosCondSmall

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class GumBoltAtlasCRBMCNN(GumBoltCaloCRBM):
    """
    GumBolt
    """

    def __init__(self, **kwargs):
        super(GumBoltAtlasCRBMCNN, self).__init__(**kwargs)
        self._model_type = "GumBoltAtlasCRBMCNN"
        self._bce_loss = BCEWithLogitsLoss(reduction="none")
        # self._energy_activation_fct = LeakyReLU(0.2) # <--- 0.02
        self._inference_energy_activation_fct = ReLU()
        
    
    def _training_activation_fct(self, slope):
        return LeakyReLU(slope)

    def create_networks(self):
        """
        - Overrides _create_networks in discreteVAE.py

        """
        logger.debug("Creating Network Structures")
        self.encoder=self._create_encoder()
        self.prior=self._create_prior()
        self.decoder=self._create_decoder()
        self.sampler = self._create_sampler(rbm=self.prior)
        
    def _create_prior(self):
        """
        - Override _create_prior in discreteVAE.py
        """
        logger.debug("GumBoltCaloCRBM::_create_prior")
        num_rbm_nodes_per_layer=self._config.model.n_latent_hierarchy_lvls*self._latent_dimensions//2
        return QimeraRBM(n_visible=num_rbm_nodes_per_layer, n_hidden=num_rbm_nodes_per_layer, fullyconnected=self._config.model.fullyconnected)
        
    def _create_sampler(self, rbm=None):
        """
        - Overrides _create_sampler in discreteVAE.py
        
        Returns:
            Gibbs Sampler
        """
        logger.debug("GumBoltCaloCRBM::_create_sampler")
        return GS(batch_size=self._config.engine.rbm_batch_size,
                   RBM=self.prior,
                   n_gibbs_sampling_steps\
                       =self._config.engine.n_gibbs_sampling_steps)

    def _create_encoder(self):
        """
        - Overrides _create_encoder in GumBoltCaloCRBM.py

        Returns:
            EncoderCNN instance
        """
        logger.debug("GumBoltAtlasCRBMCNN::_create_encoder")
        if "cylencoding" in self._config.model and self._config.model.cylencoding:
            return EncoderUCNNHPosEnc(encArch=self._config.model.encodertype,
                dev = "cuda:{0}".format(self._config.gpu_list[0]),
                lz = self._config.model.lz,
                ltheta = self._config.model.ltheta,
                lr = self._config.model.lr,
                pe = self._config.model.pe,
                input_dimension=self._flat_input_size,
                n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
                n_latent_nodes=self.n_latent_nodes,
                skip_latent_layer=False,
                smoother="Gumbel",
                cfg=self._config)
        else: 
            return EncoderUCNNH(encArch=self._config.model.encodertype,
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
        self._decoder_nodes[0] = (self._decoder_nodes[0][0]+1,
                                  self._decoder_nodes[0][1])

        if self._config.model.decodertype == "Small":
            return DecoderCNNCondSmall(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct, #<--- try identity
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)
        elif self._config.model.decodertype == "SmallUnconditioned":
            return DecoderCNNUnconditioned(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct, #<--- try identity
                              num_output_nodes = self._flat_input_size,
                              cfg=self._config)
        elif self._config.model.decodertype == "SmallPosEnc":
            return DecoderCNNPosCondSmall(node_sequence=self._decoder_nodes,
                              activation_fct=self._activation_fct, #<--- try identity
                              num_output_nodes = self._flat_input_size,
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
        x, x0 = xx
        
        out.beta, out.post_logits, out.post_samples = self.encoder(x, x0, is_training, beta_smoothing_fct)
        post_samples = out.post_samples
        post_samples = torch.cat(out.post_samples, 1)
        
        output_hits, output_activations = self.decoder(post_samples, x0)
        
        out.output_hits = output_hits
        
        # TESTING UNCONDITIONED DECODER
        #zero_eng = 0 * x0
        #uncond_hits, uncond_activations = self.decoder(post_samples, zero_eng)
        #out.uncond_hits = uncond_hits

        beta = torch.tensor(self._config.model.output_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
        if self.training:
            activation_fct_annealed = self._training_activation_fct(act_fct_slope)
            out.output_activations = activation_fct_annealed(output_activations) * torch.where(x > 0, 1., 0.)
            
            # TESTING UNCONDITIONED DECODER
            #out.uncond_activations = activation_fct_annealed(uncond_activations) * torch.where(x > 0, 1., 0.)
            
        else:
            out.output_activations = self._inference_energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
            
            # TESTING UNCONDITIONED DECODER
            #out.uncond_activations = self._inference_energy_activation_fct(uncond_activations) * torch.where(x > 0, 1., 0.)
            
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
        # entropy = self._bce_loss(logits_q_z, post_zetas)
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
        rbm_visible_samples, rbm_hidden_samples = self.sampler.block_gibbs_sampling(post_zetas_vis, method=self._config.model.rbmMethod)
        rbm_vis, rbm_hid = rbm_visible_samples.detach(), rbm_hidden_samples.detach()
        neg_energy = - self.energy_exp(rbm_vis, rbm_hid)
        
        kl_loss = entropy + pos_energy + neg_energy
        return kl_loss, entropy, pos_energy, neg_energy
    
    def energy_exp(self, rbm_vis, rbm_hid):
        """
        - Compute the energy expectation value
        
        Returns:
            rbm_energy_exp_val : mean(-vis^T W hid - a^T hid - b^T vis)
        """
        logger.debug("GumBolt::energy_exp")
        
        # Broadcast W to (pcd_batchSize * nVis * nHid)
        w, vbias, hbias = self.prior.weights, self.prior.visible_bias, self.prior.hidden_bias
        w = w + torch.zeros((rbm_vis.size(0),) + w.size(), device=rbm_vis.device)
        vbias = vbias.to(rbm_vis.device)
        hbias = hbias.to(rbm_hid.device)
        
        # Prepare H, V for torch.matmul()
        # Change V.size() from (batchSize * nVis) to (batchSize * 1 * nVis)
        vis = rbm_vis.unsqueeze(2).permute(0, 2, 1)
        # Change H.size() from (batchSize * nHid) to (batchSize * nHid * 1)
        hid = rbm_hid.unsqueeze(2)
        
        batch_energy = (- torch.matmul(vis, torch.matmul(w, hid)).reshape(-1)
                        - torch.matmul(rbm_vis, vbias)
                        - torch.matmul(rbm_hid, hbias))
        
        return torch.mean(batch_energy, 0)
    
    def generate_samples_qpu(self, num_samples=64, true_energy=None):
        """
        generate_samples()
        
        Overrides generate samples in gumboltCaloV5.py
        """
        true_energies = []
        samples = []
        # Extract the RBM parameters
        crbm_weights = self.prior.weights
        crbm_vbias = self.prior.visible_bias
        crbm_hbias = self.prior.hidden_bias
        crbm_edgelist = self.prior.pruned_edge_list
        
        qubit_idxs = self.prior.visible_qubit_idxs + self.prior.hidden_qubit_idxs
        
        visible_idx_map = {visible_qubit_idx:i for i, visible_qubit_idx in enumerate(self.prior.visible_qubit_idxs)}
        hidden_idx_map = {hidden_qubit_idx:i for i, hidden_qubit_idx in enumerate(self.prior.hidden_qubit_idxs)}
        
        # Convert the RBM parameters into Ising parameters
        dwave_weights = -(crbm_weights/4.)
        dwave_vbias = -(crbm_vbias/2. + torch.sum(crbm_weights, dim=1)/4.)
        dwave_hbias = -(crbm_hbias/2. + torch.sum(crbm_weights, dim=0)/4.)
        
        dwave_weights = torch.clamp(dwave_weights, min=-2., max=1.)
        dwave_vbias = torch.clamp(dwave_vbias, min=-2., max=2.)
        dwave_hbias = torch.clamp(dwave_hbias, min=-2., max=2.)
        
        dwave_weights_np = dwave_weights.detach().cpu().numpy()
        biases = torch.cat((dwave_vbias, dwave_hbias)).detach().cpu().numpy()
        
        # Initialize the values of biases and couplers
        h = {qubit_idx:bias for qubit_idx, bias in zip(qubit_idxs, biases)}
        J = {}
        for edge in crbm_edgelist:
            if edge[0] in self.prior.visible_qubit_idxs:
                J[edge] = dwave_weights_np[visible_idx_map[edge[0]]][hidden_idx_map[edge[1]]]
            else:
                J[edge] = dwave_weights_np[visible_idx_map[edge[1]]][hidden_idx_map[edge[0]]]
        
        response = self._qpu_sampler.sample_ising(h, J, num_reads=num_samples, auto_scale=False)
        dwave_samples, dwave_energies, origSamples = self.batch_dwave_samples(response, qubit_idxs)
        dwave_samples = torch.tensor(dwave_samples, dtype=torch.float).to(crbm_weights.device)
        
        # Convert spin Ising samples to binary RBM samples
        _ZERO = torch.tensor(0., dtype=torch.float).to(crbm_weights.device)
        _MINUS_ONE = torch.tensor(-1., dtype=torch.float).to(crbm_weights.device)
        
        dwave_samples = torch.where(dwave_samples == _MINUS_ONE, _ZERO, dwave_samples)
        self.dwave_samples = dwave_samples
        
        if true_energy is None:
            true_e = torch.rand((num_samples, 1), device=crbm_weights.device).detach() * 100.
        else:
            true_e = torch.ones((num_samples, 1), device=crbm_weights.device).detach() * true_energy
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
            # if self._config.engine.modelhits:
            sample = self._inference_energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, False)
            # else:
            #     sample = self._inference_energy_activation_fct(output_activations) * torch.ones(output_hits.size(), device=output_hits.device) 
            
            # if self._config.engine.cl_lambda != 0:
            #     labels = torch.argmax(nn.Sigmoid()(self.classifier(output_hits)), dim=1)
            #     true_energies.append((torch.pow(2,labels)*256).unsqueeze(dim=1)) 
            # else:
            true_energies.append(true_e) 
            samples.append(sample)
            
        return torch.cat(true_energies, dim=0), torch.cat(samples, dim=0)

    def loss(self, input_data, fwd_out, true_energy):
        """
        - Overrides loss in gumboltCaloV5.py
        """
        logger.debug("loss")
        
        # KL Loss
        kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(fwd_out.post_logits, fwd_out.post_samples)
        
        # MSE Loss
        sigma = 2 * torch.sqrt(torch.max(input_data, torch.min(input_data[input_data>0])))
        interpolation_param = self._config.model.interpolation_param
        ae_loss = torch.pow((input_data - fwd_out.output_activations)/sigma,2) * (1 - interpolation_param + interpolation_param*torch.pow(sigma,2)) * torch.exp(self._config.model.mse_weight*input_data)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0)
        
        # BCE Hit Loss
        hit_loss = binary_cross_entropy_with_logits(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.), weight = torch.sqrt(1 + input_data), reduction='none')
        spIdx = torch.where(input_data > 0, 0., 1.).sum(dim=1) / input_data.shape[1]
        sparsity_weight = torch.exp(self._config.model.alpha - self._config.model.gamma * spIdx)
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1) * sparsity_weight, dim=0)
        
        # TESTING UNCONDITIONED DECODER
        #Unconditioned MSE Loss
        #uncond_ae_loss = torch.pow((input_data - fwd_out.uncond_activations)/sigma,2) * (1 - interpolation_param + interpolation_param*torch.pow(sigma,2)) * torch.exp(self._config.model.mse_weight*input_data)
        #uncond_ae_loss = -0.05 * torch.mean(torch.sum(uncond_ae_loss, dim=1), dim=0)
        # Unconditioned BCE Hit Loss
        #uncond_hit_loss = binary_cross_entropy_with_logits(fwd_out.uncond_hits, torch.where(input_data > 0, 1., 0.), weight = torch.sqrt(1 + input_data), reduction='none')
        #uncond_hit_loss = -0.05 * torch.mean(torch.sum(uncond_hit_loss, dim=1) * sparsity_weight, dim=0)
        
        
        return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
                "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy,}
               #"uncond_ae_loss": uncond_ae_loss, "uncond_hit_loss": uncond_hit_loss}
        
        
    def batch_dwave_samples(self, response, qubit_idxs):
        """
        sampler.sample_ising() method returns a nested SampleSet structure
        with unique samples, energies and number of occurences stored in dict 

        Extract those values and construct a batch_size * (num_vis+num_hid) numpy array

        Returns:
            batch_samples : batch_size * (num_vis+num_hid) numpy array of samples collected by the DWave sampler
            batch_energies : batch_size * 1 numpy array of energies of samples

        UPDATE: There was a bug in which the dictionary was being processed. Thus bug has been fixed in this update
        """
        samples = []
        energies = []
        origSamples = []

        for sample_info in response.data():
            origSamples.extend([sample_info[0]]*sample_info[2]) # this is the original sample
            # the first step is to reorder
            origDict = sample_info[0] # it is a dictionary {0:-1,1:1,2:-1,3:-1,4:-1 ...} 
                                      # we need to rearrange it to {0:-1,1:1,2:-1,3:-1,132:-1 ...}
            keyorder = qubit_idxs
            reorderedDict = {k: origDict[k] for k in keyorder if k in origDict} # reorder dict

            uniq_sample = list(reorderedDict.values()) # one sample
            sample_energy = sample_info[1]
            num_occurences = sample_info[2]

            samples.extend([uniq_sample]*num_occurences)
            energies.extend([sample_energy]*num_occurences)

        batch_samples = np.array(samples)
        batch_energies = np.array(energies).reshape(-1)

        return batch_samples, batch_energies, origSamples



