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

from engine.engineCaloV3 import EngineCaloV3
from utils.histHandler import HistHandler
from utils.plotting.plotCalo import plot_calo_images
from utils.hists.RBMenergyHist import generate_rbm_energy_hist

from CaloQVAE import logging
logger = logging.getLogger(__name__)

from utils.plotting.HighLevelFeatures import HighLevelFeatures as HLF
HLF_1_photons = HLF('photon', filename='/raid/javier/Datasets/CaloVAE/data/atlas/binning_dataset_1_photons.xml')
HLF_1_pions = HLF('pion', filename='/raid/javier/Datasets/CaloVAE/data/atlas/binning_dataset_1_pions.xml')

class EngineAtlas(EngineCaloV3):

    def __init__(self, cfg, **kwargs):
        logger.info("Setting up engine Atlas.")
        super(EngineAtlas, self).__init__(cfg, **kwargs)
        self.init_beta_QA()
        
    def init_beta_QA(self):
        self.beta_QA = self._config.qpu.init_beta_qa # 4.5
        self.thrsh_met = 0
        
    def beta_value(self, epoch_anneal_start, num_batches, batch_idx, epoch):
        if epoch > epoch_anneal_start:
            delta_beta = self._config.engine.beta_smoothing_fct_final - self._config.engine.beta_smoothing_fct
            delta = (self._config.engine.n_epochs * self._config.engine.annealing_percentage - epoch_anneal_start)*num_batches
            if delta_beta > 0:
                beta = min(self._config.engine.beta_smoothing_fct + delta_beta/delta * ((epoch-1)*num_batches + batch_idx), self._config.engine.beta_smoothing_fct_final)
            else:
                beta = max(self._config.engine.beta_smoothing_fct + delta_beta/delta * ((epoch-1)*num_batches + batch_idx), self._config.engine.beta_smoothing_fct_final)
        else:
            beta = self._config.engine.beta_smoothing_fct
        return beta
    
    
    def slope_act_fct_value(self, epoch_anneal_start, num_batches, batch_idx, epoch):
        # if epoch > epoch_anneal_start:
        delta_slope = self._config.engine.slope_activation_fct_final - self._config.engine.slope_activation_fct
        delta = (self._config.engine.n_epochs * self._config.engine.annealing_percentage - epoch_anneal_start)*num_batches
        if delta_slope < 0:
            slope = max(self._config.engine.slope_activation_fct + delta_slope/delta * ((epoch-1)*num_batches + batch_idx), self._config.engine.slope_activation_fct_final)
        else:
            slope = min(self._config.engine.slope_activation_fct + delta_slope/delta * ((epoch-1)*num_batches + batch_idx), self._config.engine.slope_activation_fct_final)
        # else:
            # slope = self._config.engine.slope_activation_fct
        return slope

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
                # if self._config.reducedata:
                #     in_data = self._reduceinv(in_data, true_energy, R=self.R)
                #     fwd_output.output_activations = self._reduceinv(fwd_output.output_activations, true_energy, R=self.R)
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
                    
                    batch_loss_dict["loss"] = ae_gamma*batch_loss_dict["ae_loss"] + kl_gamma*batch_loss_dict["entropy"] + kl_gamma*batch_loss_dict["pos_energy"] + kl_gamma*batch_loss_dict["neg_energy"] + batch_loss_dict["hit_loss"] 
                    
                    batch_loss_dict["loss"] = batch_loss_dict["loss"].sum()
                    batch_loss_dict["loss"].backward()
                    
                    self._optimiser.step()
                    # Trying this to free up memory on the GPU and run validation during a training epoch
                    # - hopefully backprop will work with the code above - didn't work
                    # batch_loss_dict["loss"].detach()
                else:
                    batch_loss_dict["gamma"] = 1.0
                    batch_loss_dict["epoch"] = epoch
                    
                    batch_loss_dict["loss"] = batch_loss_dict["ae_loss"] + batch_loss_dict["kl_loss"] + batch_loss_dict["hit_loss"]
                    batch_loss_dict["ahep_loss"] = batch_loss_dict["ae_loss"] + batch_loss_dict["entropy"] + batch_loss_dict["pos_energy"] + batch_loss_dict["hit_loss"]
                    batch_loss_dict["ah_loss"] = batch_loss_dict["ae_loss"] + batch_loss_dict["hit_loss"]
                    
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
                    # self._update_histograms(input_data[0]/1000, fwd_output.output_activations)
                    
                        
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
                            sample_energies, sample_data = self._model.generate_samples(num_samples=true_energy.shape[0], true_energy=true_energy)
                            self._model.sampler._batch_size = self._config.engine.rbm_batch_size
                            # sample_energies, sample_data = self._model.generate_samples()
                            if self._config.usinglayers:
                                sample_data = self.layerTo1D(sample_data)
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

                        batch_loss_dict["input"] = plot_calo_images(input_images, particle=self._config.data.particle)
                        batch_loss_dict["recon"] = plot_calo_images(recon_images, particle=self._config.data.particle)
                        batch_loss_dict["sample"] = plot_calo_images(sample_images, particle=self._config.data.particle)
            
                        
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
            self._model_creator.save_RBM_state(config_string, encoded_data_energy)
                    
            wandb.log(val_loss_dict)
            
    def _preprocess(self, input_data, label):
        """
        Preprocess the calo image data
        - Flatten the images into a 504-d vector
        - If not scaled, divide by 1000. to scale to GeV units
        - Load the data on the GPU
        """
        in_data_flat = [image.flatten(start_dim=1) for image in input_data]
        in_data = torch.cat(in_data_flat, dim=1)
                
        true_energy = label[0]
                
        # Scaled the raw data to GeV units
        if not self._config.data.scaled and not self._config.reducedata:
            in_data = in_data/1000.
                    
        in_data = in_data.to(self._device)
        true_energy = true_energy.to(self._device).float()
        
        return in_data, true_energy, in_data_flat
        # return torch.log1p((in_data/true_energy)/0.04), true_energy, in_data_flat #<------JQTM: log(1+reduced_energy/R) w/ R=0.05 for photons

    def _reduce(self, in_data, true_energy, R=0.04):
        """
        log(1+reduced_energy/R)
        """
        
        return torch.log1p((in_data/true_energy)/R)

        
    def _reduceinv(self, in_data, true_energy, R=0.04):
        """
        log(1+reduced_energy/R)
        """
        
        return (in_data.exp() - 1)*R*true_energy

    
    def _update_histograms(self, in_data, output_activations, true_energy):
        """
        Update the coffea histograms' distributions
        """
        # Samples with uniformly distributed energies - [0, 100]
        self._model.sampler._batch_size = true_energy.shape[0]
        sample_energies, sample_data = self._model.generate_samples(num_samples=true_energy.shape[0], true_energy=true_energy)
        
        if self._config.qpu.val_w_qpu:
            try:
                # self.beta_QA, _, _, _, self.thrsh_met = self._model.find_beta(self._config.qpu.num_reads, self.beta_QA, self._config.qpu.qpu_lr, self._config.qpu.qpu_iterations, 
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

        if self._config.usinglayers:
            sample_data = self.layerTo1D(sample_data)
        
        # Update the histogram
        if self._config.data.scaled:
            # Divide by 1000. to scale the data to GeV units
            in_data_t = self._data_mgr.inv_transform(in_data.detach().cpu().numpy())/1000.
            recon_data_t = self._data_mgr.inv_transform(output_activations.detach().cpu().numpy())/1000.
            sample_data_t = self._data_mgr.inv_transform(sample_data.detach().cpu().numpy())/1000.
            # self._hist_handler.update(in_data_t, recon_data_t, sample_data_t)
            sample_dwave_data_t = self._data_mgr.inv_transform(sample_dwave_data.detach().cpu().numpy())/1000.

            self._hist_handler.update(in_data_t, recon_data_t, sample_data_t, sample_dwave_data_t)  
        elif self._config.reducedata:
            in_data_t = self._reduceinv(in_data, true_energy, R=self.R)/1000
            recon_data_t = self._reduceinv(output_activations, true_energy, R=self.R)/1000
            sample_data_t = self._reduceinv(sample_data, sample_energies, R=self.R)/1000
            self._hist_handler.update(in_data_t.detach().cpu().numpy(), 
                                      recon_data_t.detach().cpu().numpy(), 
                                      sample_data_t.detach().cpu().numpy())
        else:
            self._hist_handler.update(in_data.detach().cpu().numpy(),
                                      output_activations.detach().cpu().numpy(),
                                      sample_data.detach().cpu().numpy(), sample_dwave_data.detach().cpu().numpy())
                        
        # Samples with specific energies
        conditioning_energies = self._config.engine.sample_energies
        conditioned_samples = []
        for energy in conditioning_energies:
            #Note: Should we recompute beta before every sample run?
            sample_energies, sample_data = self._model.generate_samples(num_samples=self._config.engine.n_valid_batch_size, true_energy=energy)

            if self._config.data.scaled:
                sample_data = self._data_mgr.inv_transform(sample_data.detach().cpu().numpy())/1000. 
            elif self._config.reducedata:
                sample_data = self._reduceinv(sample_data, sample_energies, R=self.R)/1000
                sample_data = sample_data.detach().cpu()
            else:
                sample_data = sample_data.detach().cpu()
                
            if type(sample_data) == torch.Tensor:
                conditioned_samples.append(sample_data)
            else:
                conditioned_samples.append(torch.tensor(sample_data))
                        
        conditioned_samples = torch.cat(conditioned_samples, dim=0).numpy()
        self._hist_handler.update_samples(conditioned_samples)
            
    def _validate(self):
        logger.debug("engineCaloV3::validate() : Running validation during a training epoch.")
        self._model.eval()
        data_loader = self.data_mgr.val_loader
        n_val_batches = self._config.engine.n_val_batches
        
        # Accumulate metrics over several batches
        epoch_loss_dict = {}
        
        # Validation loop
        with torch.set_grad_enabled(False):
            for idx in range(n_val_batches):
                input_data, label = next(iter(data_loader))
                in_data, true_energy, in_data_flat = self._preprocess(input_data, label)

                if self._config.reducedata:
                    in_data = self._reduce(in_data, true_energy, R=self.R)

                # if self._config.usinglayers:
                #     in_data = self.parseToLayer(in_data)
                
                fwd_output_v = self._model((in_data, true_energy), False)
                # if self._config.reducedata:
                #     in_data = self._reduceinv(in_data, true_energy, R=self.R)
                #     fwd_output_v.output_activations = self._reduceinv(fwd_output.output_activations, true_energy, R=self.R)
                batch_loss_dict = self._model.loss(in_data, fwd_output_v, true_energy)

                # if self._config.usinglayers:
                #     in_data = self.layerTo1D(in_data)
                #     fwd_output_v.output_activations = self.layerTo1D(fwd_output_v.output_activations)
                #     fwd_output_v.output_hits = self.layerTo1D(fwd_output_v.output_hits)
            
                # Initialize the accumulating dictionary keys and values
                if idx == 0:
                    for key in list(batch_loss_dict.keys()):
                        epoch_loss_dict[key] = batch_loss_dict[key].detach()
                # Add loss values for the current batch to accumulating dictionary
                else:
                    for key in list(batch_loss_dict.keys()):
                        epoch_loss_dict[key] += batch_loss_dict[key].detach()
                    
        # Average the accumulated loss values
        ret_loss_dict = {}
        for key in list(epoch_loss_dict.keys()):
            ret_loss_dict[key] = epoch_loss_dict[key]/n_val_batches
        
        # Reset the model state back to training
        self._model.train()
        
        # Return the loss dictionary
        return ret_loss_dict
        
    def _log_rbm_wandb(self):
        """
        Log RBM parameter values in wandb
        - Logs the histogram for the rbm parameter distributions
        - Logs the range of the rbm parameters
        """
        prior_weights = self._model.prior.weights.detach().cpu().numpy()
        prior_visible_bias = self._model.prior.visible_bias.detach().cpu().numpy()
        prior_hidden_bias = self._model.prior.hidden_bias.detach().cpu().numpy()
        
        prior_weights = prior_weights.flatten()
        prior_weights = prior_weights[prior_weights.nonzero()]
                        
        # Tracking RBM parameter distributions during training
        prior_weights_hist = np.histogram(prior_weights, range=(-2, 2), bins=512)
        prior_weights_hist_vals = np.log(prior_weights_hist[0])
        prior_weights_hist_vals = np.where(np.isinf(prior_weights_hist_vals), 0., prior_weights_hist_vals)
        prior_weights_hist = (prior_weights_hist_vals, prior_weights_hist[1])
        
        wandb.log({"prior._weights":wandb.Histogram(np_histogram=prior_weights_hist),
                   "prior._visible_bias":wandb.Histogram(prior_visible_bias),
                   "prior._hidden_bias":wandb.Histogram(prior_hidden_bias)})
                        
        # Tracking the range of RBM parameters
        wandb.log({"prior._weights_max":np.amax(prior_weights), "prior._weights_min":np.amin(prior_weights),
                   "prior._visible_bias_max":np.amax(prior_visible_bias), "prior._visible_bias_min":np.amin(prior_visible_bias),
                   "prior._hidden_bias_max":np.amax(prior_hidden_bias), "prior._hidden_bias_min":np.amin(prior_hidden_bias)})
        
    def _log_rbm_hist_wandb(self):
        """
        Log RBM parameter custom histograms in wandb
        """
        prior_weights = self._model.prior.weights.detach().cpu().numpy()
        prior_visible_bias = self._model.prior.visible_bias.detach().cpu().numpy()
        prior_hidden_bias = self._model.prior.hidden_bias.detach().cpu().numpy()
        
        prior_weights = prior_weights.flatten()
        prior_weights = prior_weights[prior_weights.nonzero()]
        
        # Tracking the range of RBM parameters
        wandb.log({"prior._weights_max":np.amax(prior_weights), "prior._weights_min":np.amin(prior_weights),
                   "prior._visible_bias_max":np.amax(prior_visible_bias), "prior._visible_bias_min":np.amin(prior_visible_bias),
                   "prior._hidden_bias_max":np.amax(prior_hidden_bias), "prior._hidden_bias_min":np.amin(prior_hidden_bias)})
        
        weights = [[weight] for weight in prior_weights]
        weights_table = wandb.Table(data=weights, columns=["weights"])
        wandb.log({"rbm_weights": wandb.plot.histogram(weights_table, "weights", title=None)})
        
    def _energy_encoded_data(self):
        partition_size=self._config.model.n_latent_nodes_per_p
        data_loader = self.data_mgr.val_loader
        energy_encoded_data = []

        # self.model.eval()
        with torch.no_grad():
            for xx in data_loader:
                in_data, true_energy, in_data_flat = self._preprocess(xx[0],xx[1])
                if self._config.reducedata:
                    in_data = self._reduce(in_data, true_energy, R=R)
                if "PRBMCNN" in self._config.model.model_type:
                    beta, post_logits, post_samples = self.model.encoder(in_data, true_energy, False)
                elif "PRBMFCN" in self._config.model.model_type:
                    enIn = torch.cat((in_data, true_energy), dim=1)
                    beta, post_logits, post_samples = self.model.encoder(enIn, False)
                post_samples = torch.cat(post_samples, 1)
                post_samples_energy = self.model.stater.energy_samples(post_samples[:,0:partition_size], post_samples[:,partition_size:2*partition_size], 
                                                         post_samples[:,2*partition_size:3*partition_size], post_samples[:,3*partition_size:4*partition_size], 1.0 )
                energy_encoded_data.append(post_samples_energy.detach().cpu())

        energy_encoded_data = torch.cat(energy_encoded_data, dim=0)
        return energy_encoded_data
        
if __name__=="__main__":
    logger.info("Willkommen!")
    engine=Engine()
    logger.info("Success!")