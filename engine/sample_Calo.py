"""
Default Engine Class for various autoencoder models.

Tested with:
- Autoencoder
"""

import torch
import os
import coffea
import yaml
import pickle
# Weights and Biases
import wandb
import numpy as np

from engine.engine import Engine
from utils.histHandler import HistHandler
from utils.plotting.plotCalo import plot_calo_images

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class Sampling(Engine):

    def __init__(self, cfg, **kwargs):
        logger.info("Setting up engine Calo.")
        super(Sampling, self).__init__(cfg, **kwargs)
        self._hist_handler = HistHandler(cfg)
        self._best_model_loss = torch.nan_to_num(torch.tensor(float('inf')))

    def fit(self, epoch, is_training=False):
        logger.debug("Fitting model. Train mode: {0}".format(is_training))
        path='../../../configs/d_wave_config.yaml'
        with open(path, 'r') as file:
            d_wave_config = yaml.safe_load(file)
        
        # Switch model between training and evaluation mode
        # Change dataloader depending on mode

        self._model.eval()
        load_dwave = 1 # means we get new dwave samples
        data_loader = self.data_mgr.val_loader
        print("\n=======\n")
        print("sampling mode ...")
        print("\n=======\n")

        val_loss_dict = {'epoch': epoch}

        num_batches = len(data_loader)
        log_batch_idx = max(num_batches//self._config.engine.n_batches_log_train, 1)
        valid_batch_idx = max(num_batches//self._config.engine.n_valid_per_epoch, 1)
        num_epochs = self._config.engine.n_batches
        num_plot_samples = self._config.engine.n_plot_samples
        total_batches = num_batches*num_epochs
        
        kl_enabled = self._config.engine.kl_enabled
        kl_annealing = self._config.engine.kl_annealing
        kl_annealing_ratio = self._config.engine.kl_annealing_ratio
        ae_enabled = self._config.engine.ae_enabled
        
        with torch.set_grad_enabled(is_training):
            try:
                synthetic_images = torch.load('synthetic_images_'+self._config.data.particle_type+'.pt')
            except FileNotFoundError:
                synthetic_images=[]
                    
            for batch_idx, (input_data, label) in enumerate(data_loader):
                self._optimiser.zero_grad()
                
                in_data, true_energy, in_data_flat = self._preprocess(input_data, label)
                
                fwd_output = self._model((in_data, true_energy), is_training)
                batch_loss_dict = self._model.loss(in_data, fwd_output)
                    

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
                        
                if self._config.engine.enable_qpu_sampling == 0:
                    sample_dwave=False
                else:
                    sample_dwave=True
                
                if batch_idx == 0 and epoch == 1 and sample_dwave==True:
                    print('Initialing Beta pre-training process')
                    for item in range(self._config.engine.init_beta_estimation_iterations):
                        self._model.generate_samples_dwave(self._config.engine.n_valid_batch_size, new_qpu_samples=1, save_dist=True)

                #if (batch_idx % log_batch_idx) == 0:
                logger.info('Batch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(epoch,
                                                                                      batch_idx,
                                                                                      len(data_loader),
                                                                                      100.*batch_idx/len(data_loader),
                                                                                      batch_loss_dict["loss"]))
                sample_data = self._update_histograms(in_data, fwd_output.output_activations, new_qpu_samples=load_dwave, sample_dwave=sample_dwave)
        

                if self._config.data.scaled:
                    in_data = torch.tensor(self._data_mgr.inv_transform(in_data.detach().cpu().numpy()))
                    recon_data = torch.tensor(self._data_mgr.inv_transform(fwd_output.output_activations.detach().cpu().numpy()))                                   
                    sample_data = torch.tensor(self._data_mgr.inv_transform(sample_data.detach().cpu().numpy()))
                else:
                    in_data = in_data*1000.
                    recon_data = fwd_output.output_activations*1000.
                    sample_data = sample_data*1000.

                    #synthetic_images=torch.cat((synthetic_images, sample_data), 0)
                synthetic_images.append(sample_data)
                input_images = []
                recon_images = []
                sample_images = []
                if (batch_idx==0) and epoch == num_epochs: # print images only at the last sampling batch
                    print("Plotting Input, Recon, Sample, images... at epoch {0}".format(epoch))
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

                    for key in batch_loss_dict.keys():
                        if key not in val_loss_dict.keys():
                            val_loss_dict[key] = batch_loss_dict[key]

            torch.save(synthetic_images, f'synthetic_images_{self._config.data.particle_type}.pt') # save particle type

        val_loss_dict = {**val_loss_dict, **self._hist_handler.get_hist_images(), **self._hist_handler.get_scatter_plots()}
        
        if self._config.save_hists:
            hist_dict = self._hist_handler.get_hist_dict()
            for key in hist_dict.keys():
                path = os.path.join(wandb.run.dir, "{0}.coffea".format(self._config.data.particle_type + "_" + str(key)))
                coffea.util.save(hist_dict[key], path)
            
        #self._hist_handler.clear()
            
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
    
    def _update_histograms(self, in_data, output_activations, new_qpu_samples=1, sample_dwave=True):
        
        """
        Update the coffea histograms' distributions
        """
        # Samples with uniformly distributed energies - [0, 100]
        if (sample_dwave==True):
            sample_energies, sample_data = self._model.generate_samples_dwave(self._config.engine.n_valid_batch_size, new_qpu_samples=new_qpu_samples)
        else:
            sample_energies, sample_data = self._model.generate_samples(self._config.engine.n_valid_batch_size)
        
        export_sample_data = sample_data

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
        conditional_energy_ratio=(conditioning_energies[1]-conditioning_energies[0])/100
        n_conditioning_samples=round(self._config.engine.n_valid_batch_size*conditional_energy_ratio)
        if (sample_dwave==True):
            #print("new_qpu_samples is HC (in conditional energy): {0}".format(0))
            sample_energies, sample_data = self._model.generate_samples_dwave(self._config.engine.n_valid_batch_size, conditioning_energies, new_qpu_samples=0)
        else:
            sample_energies, sample_data = self._model.generate_samples(n_conditioning_samples, conditioning_energies)
        sample_data = self._data_mgr.inv_transform(sample_data.detach().cpu().numpy())/1000. if self._config.data.scaled else sample_data.detach().cpu().numpy()

        self._hist_handler.update_samples(sample_data)
        return export_sample_data

        
if __name__=="__main__":
    logger.info("Willkommen!")
    engine=Engine()
    logger.info("Success!")