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

from engine.engine import Engine
from utils.histHandler import HistHandler
from utils.plotting.plotCalo import plot_calo_images

from DiVAE import logging
logger = logging.getLogger(__name__)

class EngineCaloV3(Engine):

    def __init__(self, cfg, **kwargs):
        logger.info("Setting up engine Calo.")
        super(EngineCaloV3, self).__init__(cfg, **kwargs)
        self._hist_handler = HistHandler(cfg)
        self._best_model_loss = torch.nan_to_num(torch.tensor(float('inf')))

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
        total_batches = num_batches*num_epochs
        
        kl_enabled = self._config.engine.kl_enabled
        kl_annealing = self._config.engine.kl_annealing
        kl_annealing_ratio = self._config.engine.kl_annealing_ratio
        ae_enabled = self._config.engine.ae_enabled
        
        with torch.set_grad_enabled(is_training):
            for batch_idx, (input_data, label) in enumerate(data_loader):
                self._optimiser.zero_grad()
                
                in_data, true_energy, in_data_flat = self._preprocess(input_data, label)
                
                fwd_output=self._model((in_data, true_energy), is_training)
                batch_loss_dict = self._model.loss(in_data, fwd_output)
                    
                if is_training:
                    gamma = min((((epoch-1)*num_batches)+(batch_idx+1))/(total_batches*kl_annealing_ratio), 1.0)
                    if kl_enabled:
                        if kl_annealing:
                            kl_gamma = gamma
                        else:
                            kl_gamma = 1.
                    else:
                        kl_gamma = 0.
                        
                    if ae_enabled:
                        ae_gamma = 1.
                    else:
                        ae_gamma = 0.
                        
                    batch_loss_dict["gamma"] = kl_gamma
                    batch_loss_dict["epoch"] = gamma*num_epochs
                    if "hit_loss" in batch_loss_dict.keys():
                        batch_loss_dict["loss"] = ae_gamma*batch_loss_dict["ae_loss"] + kl_gamma*batch_loss_dict["kl_loss"] + batch_loss_dict["hit_loss"]
                    else:
                        batch_loss_dict["loss"] = ae_gamma*batch_loss_dict["ae_loss"] + kl_gamma*batch_loss_dict["kl_loss"]
                    batch_loss_dict["loss"].backward()
                    self._optimiser.step()
                    # Trying this to free up memory on the GPU and run validation during a training epoch
                    # - hopefully backprop will work with the code above - didn't work
                    # batch_loss_dict["loss"].detach()
                else:
                    batch_loss_dict["gamma"] = 1.0
                    batch_loss_dict["epoch"] = epoch
                    if "hit_loss" in batch_loss_dict.keys():
                        batch_loss_dict["loss"] = batch_loss_dict["ae_loss"] + batch_loss_dict["kl_loss"] + batch_loss_dict["hit_loss"]
                    else:
                        batch_loss_dict["loss"] = batch_loss_dict["ae_loss"] + batch_loss_dict["kl_loss"]
                    for key, value in batch_loss_dict.items():
                        try:
                            val_loss_dict[key] += value
                        except KeyError:
                            val_loss_dict[key] = value
                        
                    self._update_histograms(in_data, fwd_output.output_activations)
                    
                if mode == "train" and (batch_idx % valid_batch_idx) == 0:
                    valid_loss_dict = self._validate()
                    
                    if "hit_loss" in valid_loss_dict.keys():
                        valid_loss_dict["loss"] = valid_loss_dict["ae_loss"] + valid_loss_dict["kl_loss"] + valid_loss_dict["hit_loss"]
                    else:
                        valid_loss_dict["loss"] = valid_loss_dict["ae_loss"] + valid_loss_dict["kl_loss"]
                        
                    # Check the loss over the validation set is 
                    if valid_loss_dict["loss"] < self._best_model_loss:
                        self._best_model_loss = valid_loss_dict["loss"]
                        # Save the best model here
                        config_string = "_".join(str(i) for i in [self._config.model.model_type,
                                                                  self._config.data.data_type,
                                                                  self._config.tag, "best"])
                        self._model_creator.save_state(config_string)
                        
                if (batch_idx % log_batch_idx) == 0:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(epoch,
                                                                                          batch_idx,
                                                                                          len(data_loader),
                                                                                          100.*batch_idx/len(data_loader),
                                                                                          batch_loss_dict["loss"]))
                    
                    if (batch_idx % (num_batches//2)) == 0:
                        if self._config.data.scaled:
                            in_data = torch.tensor(self._data_mgr.inv_transform(in_data.detach().cpu().numpy()))
                            recon_data = torch.tensor(self._data_mgr.inv_transform(fwd_output.output_activations.detach().cpu().numpy()))
                            sample_energies, sample_data = self._model.generate_samples()
                            sample_data = torch.tensor(self._data_mgr.inv_transform(sample_data.detach().cpu().numpy()))
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
                        
                        batch_loss_dict["input"] = plot_calo_images(input_images)
                        batch_loss_dict["recon"] = plot_calo_images(recon_images)
                        batch_loss_dict["sample"] = plot_calo_images(sample_images)
                        
                        if not is_training:
                            for key in batch_loss_dict.keys():
                                if key not in val_loss_dict.keys():
                                    val_loss_dict[key] = batch_loss_dict[key]
                        
                    if is_training:
                        wandb.log(batch_loss_dict)
                        if (batch_idx % (num_batches//100)) == 0:
                            #self._log_rbm_wandb()
                            pass
                        
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
        if not self._config.data.scaled:
            in_data = in_data/1000.
                    
        in_data = in_data.to(self._device)
        true_energy = true_energy.to(self._device).float()
        
        return in_data, true_energy, in_data_flat
    
    def _update_histograms(self, in_data, output_activations):
        """
        Update the coffea histograms' distributions
        """
        # Samples with uniformly distributed energies - [0, 100]
        sample_energies, sample_data = self._model.generate_samples(self._config.engine.n_valid_batch_size)
        
        # Update the histogram
        if self._config.data.scaled:
            # Divide by 1000. to scale the data to GeV units
            in_data_t = self._data_mgr.inv_transform(in_data.detach().cpu().numpy())/1000.
            recon_data_t = self._data_mgr.inv_transform(output_activations.detach().cpu().numpy())/1000.
            sample_data_t = self._data_mgr.inv_transform(sample_data.detach().cpu().numpy())/1000.
            self._hist_handler.update(in_data_t, recon_data_t, sample_data_t)                  
        else:
            self._hist_handler.update(in_data.detach().cpu().numpy(),
                                      output_activations.detach().cpu().numpy(),
                                      sample_data.detach().cpu().numpy())
                        
        # Samples with specific energies
        conditioning_energies = self._config.engine.sample_energies
        conditioned_samples = []
        for energy in conditioning_energies:
            sample_energies, sample_data = self._model.generate_samples(self._config.engine.n_valid_batch_size, energy)
            sample_data = self._data_mgr.inv_transform(sample_data.detach().cpu().numpy())/1000. if self._config.data.scaled else sample_data.detach().cpu().numpy()
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
                
                fwd_output = self._model((in_data, true_energy), False)
                batch_loss_dict = self._model.loss(in_data, fwd_output)
            
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
        
if __name__=="__main__":
    logger.info("Willkommen!")
    engine=Engine()
    logger.info("Success!")