"""
Default Engine Class for various autoencoder models.

Tested with:
- Autoencoder
"""

import torch
import os
import coffea

# Weights and Biases
import wandb
import numpy as np

from utils.histHandler import HistHandler
from utils.plotting.plotCalo import plot_calo_images
from utils.hists.RBMenergyHist import generate_rbm_energy_hist
from engine.engineAtlas import EngineAtlas
from models.networks.Discriminator import Discriminator

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class EngineAtlasDisc(EngineAtlas):

    def __init__(self, cfg, **kwargs):
        logger.info("Setting up engine Atlas with Discriminator.")
        super(EngineAtlasDisc, self).__init__(cfg, **kwargs)
        self.critic = Discriminator()
        self.optimiser_c = torch.optim.Adam(self.critic.parameters(),
                                        lr=self._config.engine.learning_rate)
        self.critic_2 = Discriminator()
        self.optimiser_c_2 = torch.optim.Adam(self.critic_2.parameters(),
                                        lr=self._config.engine.learning_rate)
        
    # def loss_wgan_1(self, input_data, fwd_out):
    def loss_wgan_1(self, critic, real_c, fake_c, true_energy):
        """
        Wasserstein GAN
        """
        # real_c = torch.zeros_like(input_data)
        # fake_c = input_data - fwd_out.output_activations
        # real_c = input_data
        # fake_c = fwd_out.output_activations
        
        f_real = critic(real_c, true_energy)
        f_fake = critic(fake_c, true_energy)
        
        l_critic = - (f_real.mean() - f_fake.mean())
        
        return l_critic
    
    # def wgan_gp_critic_loss(self, input_data, fwd_out, lambda_gp=10):
    def wgan_gp_critic_loss(self, critic, real, fake, true_energy, lambda_gp=10):
        # real = input_data
        # fake = fwd_out.output_activations
        # 1. Critic outputs
        crit_real = critic(real, true_energy).mean()   # mean over batch
        crit_fake = critic(fake, true_energy).mean()

        # 2. Wasserstein loss (negative of the difference)
        wloss = -(crit_real - crit_fake)

        # 3. Compute gradient penalty
        gp = self.gradient_penalty(critic, real, fake, true_energy, lambda_gp)

        # 4. Combine
        loss_critic = wloss + gp
        return loss_critic #, crit_real, crit_fake, gp

        
    # def loss_wgan_2(self, input_data, fwd_out):
    def loss_wgan_2(self, critic, fake_c, true_energy):
        """
        Wasserstein GAN
        """
        # fake_c = input_data - fwd_out.output_activations
        # fake_c = fwd_out.output_activations
        f_fake = critic(fake_c, true_energy)
        
        l_gen = -f_fake.mean()
        
        return l_gen
    
    def gradient_penalty(self, critic, real_data, fake_data, true_energy, lambda_gp=10):
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1).to(real_data.device)
        eps = eps.expand_as(real_data)
        
        # Interpolation between real data and fake data.
        interpolation = eps * real_data + (1 - eps) * fake_data
        
        # get logits for interpolated images
        interp_logits = critic(interpolation, true_energy)
        grad_outputs = torch.ones_like(interp_logits)
        
        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2) * lambda_gp
        
    def fit(self, epoch, is_training=True, mode="train"):
        logger.debug("Fitting model. Train mode: {0}".format(is_training))

        # Switch model between training and evaluation mode
        # Change dataloader depending on mode
        if is_training:
            self._model.train()
            data_loader = self.data_mgr.train_loader
        else:
            self._model.eval()
            if mode == "validate":
                data_loader = self.data_mgr.val_loader
            elif mode == "test":
                data_loader = self.data_mgr.test_loader
            val_loss_dict = {'epoch': epoch}

        num_batches = len(data_loader)
        log_batch_idx = max(num_batches//self._config.engine.n_batches_log_train, 1)
        valid_batch_idx = max(num_batches//self._config.engine.n_valid_per_epoch, 1)
        num_epochs = self._config.engine.n_epochs
        num_plot_samples = self._config.engine.n_plot_samples
        
        kl_enabled = self._config.engine.kl_enabled
        kl_annealing = self._config.engine.kl_annealing
        kl_annealing_ratio = self._config.engine.kl_annealing_ratio
        ae_enabled = self._config.engine.ae_enabled
        epoch_anneal_start = self._config.engine.epoch_annealing_start
        total_batches = num_batches*(num_epochs-epoch_anneal_start+1)
        self.R = self._config.engine.r_param
        # cl_lambda = self._config.engine.cl_lambda
        
        with torch.set_grad_enabled(is_training):
            for batch_idx, (input_data, label) in enumerate(data_loader):
                self._optimiser.zero_grad()
                
                in_data, true_energy, in_data_flat = self._preprocess(input_data, label)
                
                if self._config.reducedata:
                    in_data = self._reduce(in_data, true_energy, R=self.R)

                    
                if self._config.engine.beta_smoothing_fct_anneal and self._config.load_state == 0:
                    beta_smoothing_fct = self.beta_value(epoch_anneal_start, num_batches, batch_idx, epoch)
                else:
                    beta_smoothing_fct = self._config.engine.beta_smoothing_fct_final
                    
                
                if self._config.engine.slope_activation_fct_anneal and self._config.load_state == 0:
                    slope_act_fct = self.slope_act_fct_value(epoch_anneal_start, num_batches, batch_idx, epoch)
                else:
                    slope_act_fct = self._config.engine.slope_activation_fct_final
                
                fwd_output = self._model((in_data, true_energy), is_training, beta_smoothing_fct, slope_act_fct)
               
                batch_loss_dict = self._model.loss(in_data, fwd_output, true_energy)

                    
                if is_training:
                    if epoch >= epoch_anneal_start:
                        # gamma = min((((epoch-epoch_anneal_start)*num_batches)+(batch_idx+1))/(total_batches*kl_annealing_ratio), self._config.engine.kl_gamma_max)
                        gamma = 1
                    else:
                        gamma = 0
                    if kl_enabled:
                        if kl_annealing:
                            kl_gamma = gamma
                        else:
                            kl_gamma = 1.
                    else:
                        kl_gamma = 0.
                        
                    ae_gamma = 1. if ae_enabled else 0.
                        
                    batch_loss_dict["gamma"] = kl_gamma
                    batch_loss_dict["LeakyReLUSlope"] = slope_act_fct
                    batch_loss_dict["beta"] = beta_smoothing_fct
                    batch_loss_dict["epoch"] = gamma*num_epochs
                    
                    batch_loss_dict["ahep_loss"] = batch_loss_dict["ae_loss"] + batch_loss_dict["entropy"] + batch_loss_dict["pos_energy"]  + batch_loss_dict["hit_loss"]
                    batch_loss_dict["ah_loss"] = batch_loss_dict["ae_loss"] + batch_loss_dict["hit_loss"]
                    
                    if 'exact_rbm_grad' in self._config.keys() and self._config.exact_rbm_grad:
                        batch_loss_dict["loss"] = ae_gamma*batch_loss_dict["ae_loss"] + kl_gamma*batch_loss_dict["entropy"] + kl_gamma*batch_loss_dict["pos_energy"] + batch_loss_dict["hit_loss"] 
                        if self._config.rbm_grad_centered:
                            self.model.sampler.gradient_rbm_centered(fwd_output.post_samples, self._config.model.n_latent_nodes_per_p, self._config.model.rbmMethod )
                        else:
                            self.model.sampler.gradient_rbm(fwd_output.post_samples, self._config.model.n_latent_nodes_per_p, self._config.model.rbmMethod )
                        self.model.sampler.update_params()
                    else:
                        batch_loss_dict["loss"] = ae_gamma*batch_loss_dict["ae_loss"] + kl_gamma*batch_loss_dict["entropy"] + kl_gamma*batch_loss_dict["pos_energy"] + kl_gamma*batch_loss_dict["neg_energy"] + batch_loss_dict["hit_loss"] 
                        
                    
                    if self._config.engine.discriminator and epoch > 2 and epoch < self._config.engine.epoch_freeze:
                        fake = in_data - fwd_output.output_activations
                        batch_loss_dict["generator"] = self.loss_wgan_2(self.critic, fake, true_energy)
                        batch_loss_dict["loss"] = batch_loss_dict["loss"] + batch_loss_dict["generator"]
                        ############critic 2    
                        fake = fwd_output.output_activations
                        batch_loss_dict["generator_2"] = self.loss_wgan_2(self.critic_2, fake, true_energy)
                        batch_loss_dict["loss"] = batch_loss_dict["loss"] + batch_loss_dict["generator_2"]
                        ############################
                        
                        batch_loss_dict["loss"] = batch_loss_dict["loss"].sum()
                        batch_loss_dict["loss"].backward()
                        self._optimiser.step()
                        
                        for _ in range(self._config.engine.n_critic):
                            self._optimiser_c.zero_grad()
                            fwd_output = self._model((in_data, true_energy), is_training, beta_smoothing_fct, slope_act_fct)
                            real = torch.zeros_like(in_data)
                            fake = in_data - fwd_output.output_activations
                            # batch_loss_dict["critic"] = self.loss_wgan_1(real, fake, , true_energy)
                            batch_loss_dict["critic"] = self.wgan_gp_critic_loss(self.critic, real, fake, true_energy, self._config.engine.gp_l)
                            batch_loss_dict["critic"].backward()
                            self._optimiser_c.step()
                            # if epoch % 10 == 0 and epoch > 5:
                            #     for p in self.critic.parameters():
                            #         p.data.clamp_(-self._config.engine.clip_value, self._config.engine.clip_value)
                            
                        
                        for _ in range(self._config.engine.n_critic):
                            self._optimiser_c_2.zero_grad()
                            fwd_output = self._model((in_data, true_energy), is_training, beta_smoothing_fct, slope_act_fct)
                            real = in_data
                            fake = fwd_output.output_activations
                            batch_loss_dict["critic_2"] = self.wgan_gp_critic_loss(self.critic_2, real, fake, true_energy, self._config.engine.gp_l)
                            batch_loss_dict["critic_2"].backward()
                            self._optimiser_c_2.step()
                    else:
                        batch_loss_dict["loss"] = batch_loss_dict["loss"].sum()
                        try:
                            batch_loss_dict["loss"].backward()
                        except:
                            pass
                        
                    self._optimiser.step()
                        
                else:
                    batch_loss_dict["gamma"] = 1.0
                    batch_loss_dict["epoch"] = epoch
                    
                    batch_loss_dict["loss"] = batch_loss_dict["ae_loss"] + batch_loss_dict["kl_loss"] + batch_loss_dict["hit_loss"]
                    
                    
                    for key, value in batch_loss_dict.items():
                        try:
                            val_loss_dict[key] += value
                        except KeyError:
                            val_loss_dict[key] = value
                            
                    if self._config.qpu.val_w_qpu and batch_idx == 0 and mode == "validate":
                        try:
                            self.beta_QA, _, _, _, self.thrsh_met = self._model.find_beta(self._config.qpu.num_reads, self.beta_QA, self._config.qpu.qpu_lr, self._config.qpu.qpu_iterations, 
                                                                                                              self._config.qpu.power, self._config.qpu.method, True, self._config.qpu.thrs_const, self._config.qpu.adaptive)
                            # if self.thrsh_met == 0:
                            #     logger.warn("We regret to inform you that the threshold was not met. The samples will be classically generated.")
                            #     sample_dwave_energies, sample_dwave_data = sample_energies, sample_data
                            # else:
                            #     sample_dwave_energies, sample_dwave_data = self._model.generate_samples_qpu(num_samples=true_energy.shape[0], true_energy=true_energy, beta=1.0/self.beta_QA)
                        except:
                            logger.warn("Unable to use QPU :'( . You probably ran outta $$. We'll use classical sampling instead")
                            # sample_dwave_energies, sample_dwave_data = sample_energies, sample_data
                    # else:
                    #     sample_dwave_energies, sample_dwave_data = sample_energies, sample_data
                    
                    self._update_histograms(in_data, fwd_output.output_activations, true_energy)
                    
                        
                if (batch_idx % log_batch_idx) == 0:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(epoch,
                                                                                          batch_idx,
                                                                                          len(data_loader),
                                                                                          100.*batch_idx/len(data_loader),
                                                                                          batch_loss_dict["loss"].sum()))
                    
                    if (batch_idx % (num_batches//2)) == 0:
                        if self._config.data.scaled:
                            in_data = torch.tensor(self._data_mgr.inv_transform(in_data.detach().cpu().numpy()))
                            recon_data = torch.tensor(self._data_mgr.inv_transform(fwd_output.output_activations.detach().cpu().numpy()))
                            self._model.sampler._batch_size = true_energy.shape[0]
                            sample_energies, sample_data = self._model.generate_samples(num_samples=true_energy.shape[0], true_energy=true_energy)
                            
                            if self._config.qpu.val_w_qpu and batch_idx == 0 and mode == "validate" and False:
                                try:
                                    # self.beta_QA, _, _, _, self.thrsh_met = self._model.find_beta(self._config.qpu.num_reads, self._config.qpu.num_reads, self.beta_QA, self._config.qpu.qpu_lr, self._config.qpu.qpu_iterations, 
                                                                                                  # self._config.qpu.power, self._config.qpu.method, True, self._config.qpu.thrs_const, self._config.qpu.adaptive)
                                    if self.thrsh_met == 0:
                                        logger.warn("We regret to inform you that the threshold was not met. The samples will be classically generated.")
                                        sample_dwave_energies, sample_dwave_data = sample_energies, sample_data
                                    else:
                                        sample_dwave_energies, sample_dwave_data = self._model.generate_samples_qpu(num_samples=true_energy.shape[0], true_energy=true_energy, beta=1.0/self.beta_QA)
                                except:
                                    logger.warn("Unable to use QPU :'( . You probably ran outta $$. We'll use classical sampling instead")
                                    sample_dwave_energies, sample_dwave_data = sample_energies, sample_data
                            else:
                                sample_dwave_energies, sample_dwave_data = sample_energies, sample_data

                            self._model.sampler._batch_size = self._config.engine.rbm_batch_size
                            # sample_energies, sample_data = self._model.generate_samples()
                            sample_data = torch.tensor(self._data_mgr.inv_transform(sample_data.detach().cpu().numpy()))
                        elif self._config.reducedata:
                            in_data = self._reduceinv(in_data, true_energy, R=self.R)
                            recon_data = self._reduceinv(fwd_output.output_activations, true_energy, R=self.R)
                            self._model.sampler._batch_size = true_energy.shape[0]
                            # sample_energies, sample_data = self._model.generate_samples(num_samples=true_energy.shape[0], true_energy=true_energy)
                            sample_energies, sample_data = self._model.generate_samples_cond(num_samples=true_energy.shape[0], true_energy=true_energy)
                            self._model.sampler._batch_size = self._config.engine.rbm_batch_size
                            # sample_energies, sample_data = self._model.generate_samples()
                            # if self._config.usinglayers:
                                # sample_data = self.layerTo1D(sample_data)
                            sample_data = self._reduceinv(sample_data, sample_energies, R=self.R)
                        else:
                            # Multiply by 1000. to scale to MeV
                            in_data = in_data*1000.
                            recon_data = fwd_output.output_activations*1000.
                            sample_energies, sample_data = self._model.generate_samples()
                            sample_data = sample_data*1000.
                            
                        input_images = []
                        recon_images = []
                        sample_images = []

                        start_index = 0
                        for layer, layer_data_flat in enumerate(in_data_flat):
                            input_image = in_data[:num_plot_samples, start_index:start_index+layer_data_flat.size(1)]
                            recon_image = recon_data[:num_plot_samples, start_index:start_index+layer_data_flat.size(1)]
                            sample_image = sample_data[:num_plot_samples, start_index:start_index+layer_data_flat.size(1)]
                            
                            start_index += layer_data_flat.size(1)
                            
                            input_image = input_image.reshape((-1,) + input_data[layer].size()[1:]).detach().cpu().numpy()
                            recon_image = recon_image.reshape((-1,) + input_data[layer].size()[1:]).detach().cpu().numpy()
                            sample_image = sample_image.reshape((-1,) + input_data[layer].size()[1:]).detach().cpu().numpy()
                            
                            input_images.append(input_image)
                            recon_images.append(recon_image)
                            sample_images.append(sample_image)

#                         logger.info(input_images[0].shape)

                        # batch_loss_dict["input"] = plot_calo_images(input_images, particle=self._config.data.particle)
                        # batch_loss_dict["recon"] = plot_calo_images(recon_images, particle=self._config.data.particle)
                        # batch_loss_dict["sample"] = plot_calo_images(sample_images, particle=self._config.data.particle)
                        batch_loss_dict["input"] = plot_calo_images(input_images, self.HLF)
                        batch_loss_dict["recon"] = plot_calo_images(recon_images, self.HLF)
                        batch_loss_dict["sample"] = plot_calo_images(sample_images, self.HLF)
            
                        
                        if not is_training:
                            for key in batch_loss_dict.keys():
                                if key not in val_loss_dict.keys():
                                    val_loss_dict[key] = batch_loss_dict[key]
                        
                    if is_training:
                        wandb.log(batch_loss_dict)
                        if (batch_idx % max((num_batches//100), 1)) == 0:
                            #self._log_rbm_wandb()
                            pass

                if mode == "train" and (batch_idx % valid_batch_idx) == 0:
                    valid_loss_dict = self._validate()
                    
                    if "hit_loss" in valid_loss_dict.keys():
                        valid_loss_dict["loss"] = valid_loss_dict["ae_loss"] + valid_loss_dict["kl_loss"] + valid_loss_dict["hit_loss"]
                        valid_loss_dict["ahep_loss"] = valid_loss_dict["ae_loss"] + valid_loss_dict["entropy"] + valid_loss_dict["pos_energy"] + valid_loss_dict["hit_loss"]
                        valid_loss_dict["ah_loss"] = valid_loss_dict["ae_loss"] + valid_loss_dict["hit_loss"]
                    else:
                        valid_loss_dict["loss"] = valid_loss_dict["ae_loss"] + valid_loss_dict["kl_loss"]
                        valid_loss_dict["ahep_loss"] = valid_loss_dict["ae_loss"] + valid_loss_dict["entropy"] + valid_loss_dict["pos_energy"]
                        valid_loss_dict["ah_loss"] = valid_loss_dict["ae_loss"] + valid_loss_dict["hit_loss"]
                    # wandb.log(val_loss_dict)
                    # Check the loss over the validation set is 
                    # if valid_loss_dict["loss"].sum() < self._best_model_loss:
                    if valid_loss_dict["ah_loss"].sum() < self._best_model_loss:
                        self._best_model_loss = valid_loss_dict["ah_loss"].sum()
                        # Save the best model here
                        config_string = "_".join(str(i) for i in [self._config.model.model_type,
                                                                  self._config.data.data_type,
                                                                  self._config.tag, f'best'])
                        self._model_creator.save_state(config_string)
                        
                        
        if not is_training:
            val_loss_dict = {**val_loss_dict, **self._hist_handler.get_hist_images(), **self._hist_handler.get_scatter_plots()}
            
            if self._config.save_hists:
                hist_dict = self._hist_handler.get_hist_dict()
                for key in hist_dict.keys():
                    path = os.path.join(wandb.run.dir, "{0}.coffea".format(self._config.data.particle_type + "_" + str(key)))
                    coffea.util.save(hist_dict[key], path)
                
            self._hist_handler.clear()
                
            # Average the validation loss values over the validation set
            # Modify the logging keys to prefix with 'val_'
            for key in list(val_loss_dict.keys()):
                try:
                    val_loss_dict['val_' + str(key)] = val_loss_dict[key]/num_batches
                    val_loss_dict.pop(key)
                except TypeError:
                    val_loss_dict['val_' + str(key)] = val_loss_dict[key]
                    val_loss_dict.pop(key)
                    
            rbm_energy_hist = []
            rbm_energy_hist.append(generate_rbm_energy_hist(self, self._config.model, self.data_mgr.val_loader))
            val_loss_dict["RBM energy"] = rbm_energy_hist
            config_string = f'RBM_{epoch}_{batch_idx}'
            encoded_data_energy = self._energy_encoded_data()
            # if epoch % 1000 == 0:
            self._model_creator.save_RBM_state(config_string, encoded_data_energy)
                    
            wandb.log(val_loss_dict)
        
if __name__=="__main__":
    logger.info("Willkommen!")
    engine=Engine()
    logger.info("Success!")