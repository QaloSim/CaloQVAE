"""
GumBolt DVAE with Pegasus RBM prior
"""
import torch

# CaloQVAE.models imports
from CaloQVAE.models.rbm import pegasusRBM
from CaloQVAE.models.samplers import pgbs
from models.autoencoders import gumboltCaloV6 as gcv6


class GumBoltCaloPRBM(gcv6.GumBoltCaloV6):
    """
    GumBoltDVAE with energy conditioning and a 4-partite PegasusRBM prior
    """

    def __init__(self, **kwargs):
        super(GumBoltCaloPRBM, self).__init__(**kwargs)
        self._model_type = 'GumBoltCaloPRBM'

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
        w_dict = self.prior.weight_dict
        b_dict = self.prior.bias_dict

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

    def generate_samples(self, n_samples: int = 128, true_energy: int = -1):
        """Generate data samples by decoding RBM samples

        :param n_samples (int): No. of data samples to generate in one shot
        :param true_energy (int): Default None, Incident energy of the particle

        :return true_energies (torch.Tensor): Incident energies of the particle
        for each sample (n_samples,)
        :return samples (torch.Tensor): Data samples, (n_samples, *)
        """
        n_iter = max(n_samples//self.sampler.batch_size, 1)
        true_es, samples = [], []

        for _ in range(n_iter):
            p0_state, p1_state, p2_state, p3_state \
                = self.sampler.block_gibbs_sampling()

            if true_energy == -1:
                # true_e ~ U[1, 100]
                true_e = (torch.rand((p0_state.size(0), 1),
                                     device=p0_state.device) * 99.) + 1.
            else:
                # true_e = true_energy
                true_e = torch.ones((p0_state.size(0), 1),
                                    device=p0_state.device) * true_energy
            prior_samples = torch.cat([p0_state, p1_state, p2_state, p3_state,
                                       true_e], dim=1)

            hits, activations = self.decoder(prior_samples)
            beta = torch.tensor(self._config.model.beta_smoothing_fct,
                                dtype=torch.float, device=hits.device)
            sample = self._energy_activation_fct(activations) \
                * self._hit_smoothing_dist_mod(hits, beta, False)

            true_es.append(true_e)
            samples.append(sample)

        return torch.cat(true_es, dim=0), torch.cat(samples, dim=0)
