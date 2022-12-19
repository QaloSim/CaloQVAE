"""
Default Engine Class for various autoencoder models.

Tested with:
- Autoencoder
"""

import torch
import os
import coffea
import yaml

# Weights and Biases
import wandb
import numpy as np

from engine.engine import Engine
from utils.histHandler import HistHandler
from utils.plotting.plotCalo import plot_calo_images

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class EngineCaloQV1(Engine):

    def __init__(self, cfg, **kwargs):
        logger.info("Setting up engine Calo.")
        super(EngineCaloQV1, self).__init__(cfg, **kwargs)
        self._hist_handler = HistHandler(cfg)
        self._best_model_loss = torch.nan_to_num(torch.tensor(float('inf')))

    def fit(self, epoch, is_training=True, mode="train"):
        logger.debug("Fitting model. Train mode: {0}".format(is_training))
        path='../../../configs/d_wave_config.yaml'
        with open(path, 'r') as file:
            d_wave_config = yaml.safe_load(file)
        sampling_epoches_list=list(d_wave_config['sampling_epoches'])
        
        # Switch model between training and evaluation mode
        # Change dataloader depending on mode
        if is_training:
            load_dwave = 0 # means we do not get new dwave samples
            self._model.train()
            data_loader = self.data_mgr.train_loader
            print("\n=======\n")
            print("Training mode ...")
            print("\n=======\n")
        else:
            self._model.eval()
            if mode == "validate":
                load_dwave = 1 # means we get new dwave samples
                data_loader = self.data_mgr.val_loader
                print("\n=======\n")
                print("validation mode ...")
                print("\n=======\n")
            elif mode == "test":
                load_dwave = 0 # means we do not get new dwave samples
                data_loader = self.data_mgr.test_loader
                print("\n=======\n")
                print("Testing mode ...")
                print("\n=======\n")
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
            if mode == "validate":
                print("\nvalidate mode in the beginning\n")
                synthetic_images = []
            for batch_idx, (input_data, label) in enumerate(data_loader):
                self._optimiser.zero_grad()
                
                in_data, true_energy, in_data_flat = self._preprocess(input_data, label)
                
                fwd_output = self._model((in_data, true_energy), is_training)
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
                        
                    ae_gamma = 1. if ae_enabled else 0.
                        
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
                            
                    if (epoch not in sampling_epoches_list):
                        sample_dwave=False
                    else:
                        sample_dwave=True
                    self._update_histograms(in_data, fwd_output.output_activations, new_qpu_samples=load_dwave, sample_dwave=sample_dwave)
                    """
                    This ensures that load_dwave is 1 ONLY in the very first batch ...
                    even if load_dwave is 0, I will go to QPU as generate_samples_dwave is called only depending on
                    sample_dwave and not on load dwave...
                    essentially if i do not make my load_dwave to be 0, i wll be sampling at each BATCH. which seems unncessary as I do expect to get the same samples out in the end. 
                    this again re-iterates my point that generating new ssmples using BGS or dwave does not really have ANY effect. rather the random true_energy
                    helps to ensure that we get different variety of samples. 
                    this is beacuse in each validation BATCH step, my J and h ARE THE SAME!!!!
                    THERFORE i have say 1024 samples from dwave. I reuse the sampe 1024 samples in all my batch step. as I have 10 batches I expect to get 1024*10 different outputs. 
                    q: what are thr dwave/bgs samples used for/
                    """
                    load_dwave = 0 # if load_dwave was 1, it is now changed to 0
                    
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
                    """
                    vvi :::
                    if u don't want quantum sampling, just make sampling_epochs > num_epochs (OR MAYBE FALSE OPTION?)
                    if max(m_ep_lis) > num_epoch:
                        append num_epoch in list so that we can get IMAGES AT THE VERY LAST EPOCH
                        HENCE IN SUCH A CASE SET A HANDLER TO SIMPLY SWITCH TO GENERATE_SAMPLES INSTEAD OF GEN SAMPLES DWAVE 
                    """
                    
                    if (epoch in list(self._config.engine.generate_images_at_epoch)) and mode=="validate":
                        """
                        For the listed epoch (in sampling_epochs_list), we 
                        """
                        if True:#(batch_idx  != 20):
                            
                            if self._config.data.scaled:
                                in_data = torch.tensor(self._data_mgr.inv_transform(in_data.detach().cpu().numpy()))
                                recon_data = torch.tensor(self._data_mgr.inv_transform(fwd_output.output_activations.detach().cpu().numpy()))
                                if (sample_dwave==True):
                                    """
                                    Using DWAVE samples for image generation
                                    Note that if new_qpu_samples=1 (if/else), then new samples will be requested
                                    in EACH batch. That means new samples will be requested from the latent ptior distribution
                                    in each validation step. This will help to ensure we get a good variety of samples from the latent distribution. 
                                    But this might not be necessary after all. (still I have generatd 10240 NEW SAMPLES IN TOTAL for piplus AND THAT CAN BE USED FOR CLASSIFICATION
                                    PURPOSES I GUESS.) now maybe semi-train a model for piplus and parallely try to buildup the classification
                                    """
                                    sample_energies, sample_data = self._model.generate_samples_dwave(self._config.engine.n_valid_batch_size, new_qpu_samples=1, save_dist=True)
                                else:
                                    """
                                    Using Classical RBM samples for image generation
                                    """
                                    sample_energies, sample_data = self._model.generate_samples(self._config.engine.n_valid_batch_size)                                   
                                sample_data = torch.tensor(self._data_mgr.inv_transform(sample_data.detach().cpu().numpy()))
                            else:
                                in_data = in_data*1000.
                                recon_data = fwd_output.output_activations*1000.
                                if (sample_dwave==True):
                                    sample_energies, sample_data = self._model.generate_samples_dwave(self._config.engine.n_valid_batch_size, new_qpu_samples=1, save_dist=True)
                                else:
                                    sample_energies, sample_data = self._model.generate_samples(self._config.engine.n_valid_batch_size)
                                sample_data = sample_data*1000.
                                
                            """
                            dim sample_data = 1024 * 504 (and 504 = 3*96 + 12*12 + 12+6)
                            so I can just pass in this sample_data as this sample_data gets processed further to produce images ... which I don't need.
                            In such a case, I can potentially keep num_plot_samples unchanged, 
                            thrn I can legit just draw 100,000 samples from the qpu
                            But the max num I can draw at a time is 10,000. 
                            UPDATE: SAVE THE sample_data somewhere and we will use it for classification
                            do whole classification and then maybe we can see if we can draw more. bUT FOR NOW, JUST USE CURRENT 1024 READS FOR CLASSIFICATION.
                            ** Essentially what I can do is given in the notebook ...
                            """
                            synthetic_images.append(sample_data)                            
                            input_images = []
                            recon_images = []
                            sample_images = []
                            print("sample_data SIZE is {0}\n".format(sample_data.size()))
                            ###############################################################################################################################
                            if batch_idx==0 or batch_idx==6 or batch_idx==8:
                                print("Plotting Input, Recon, Sample, images... at epoch {0}\n".format(epoch))
                                start_index = 0
                                for layer, layer_data_flat in enumerate(in_data_flat):
                                    print("start_index is {0} and layer_data_flar.size(1) is {1}\n".format(start_index, layer_data_flat.size(1)))
                                    input_image = in_data[:num_plot_samples, start_index:start_index+layer_data_flat.size(1)]
                                    recon_image = recon_data[:num_plot_samples, start_index:start_index+layer_data_flat.size(1)]
                                    sample_image = sample_data[:num_plot_samples, start_index:start_index+layer_data_flat.size(1)] # num_plot_samples is the num_gen_samples of (5,3,96) # other PARAMs don't matter
                                    print("size of sample image originally is {0}\n".format(sample_image.size()))
                                    # print("sample_image in the for loop is {0}\n".format(sample_image)) # this has the 5 images separately, while sample_image found below CONCATS THEM in another tensor ...
                                    # num_train_samples = 50 # HARDCODED
                                    # sample_image_gen = sample_data[:num_train_samples, start_index:start_index+layer_data_flat.size(1)] ## NOT SO SIMPLE **
                                    # OINT TO NOTE IS THAT SAMPLE DATA COMES FROM GENERATE_SAMPLES_DWAVE ... LET'S INSPECT THAT
                                    start_index += layer_data_flat.size(1)
                                    print("updates start index is {0}\n".format(start_index))
                                    print("input_data[layer].size() is {0}\n--\n".format(input_data[layer].size()))
                                    print("input_data[layer].size()[1:] is {0}\n".format(input_data[layer].size()[1:]))

                                    input_image = input_image.reshape((-1,) + input_data[layer].size()[1:]).detach().cpu().numpy()
                                    recon_image = recon_image.reshape((-1,) + input_data[layer].size()[1:]).detach().cpu().numpy()
                                    sample_image = sample_image.reshape((-1,) + input_data[layer].size()[1:]).detach().cpu().numpy()

                                    input_images.append(input_image)
                                    recon_images.append(recon_image)
                                    sample_images.append(sample_image)

                                    #print("sample_image array is {0}\n".format(sample_image))
                                    print("SIZE OF sample_image AFTERWARDS is {0}\n".format(np.array(sample_image).shape))
                                    #print("size of sample image originally is {0}\n".format(sample_image.size()))

                                batch_loss_dict["input"] = plot_calo_images(input_images)
                                batch_loss_dict["recon"] = plot_calo_images(recon_images)
                                batch_loss_dict["sample"] = plot_calo_images(sample_images)

                                if not is_training:
                                    for key in batch_loss_dict.keys():
                                        if key not in val_loss_dict.keys():
                                            val_loss_dict[key] = batch_loss_dict[key]
                            ###############################################################################################################################
                    
                    
                        
                    if is_training:
                        wandb.log(batch_loss_dict)
                        if (batch_idx % max((num_batches//100), 1)) == 0:
                            #self._log_rbm_wandb()
                            pass
            if mode == "validate":
                print("\nmode is validate at end\n")
                torch.save(synthetic_images, 'synthetic_images_piplus.pt')
                        
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
    
    def _update_histograms(self, in_data, output_activations, new_qpu_samples=1, sample_dwave=True):
        
        """
        Update the coffea histograms' distributions
        """
        # Samples with uniformly distributed energies - [0, 100]
        if (sample_dwave==True):
            sample_energies, sample_data = self._model.generate_samples_dwave(self._config.engine.n_valid_batch_size, new_qpu_samples=new_qpu_samples)
        else:
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
            #print("In conditioning-------------------(below)")
            #print("Sampling Classically")
            if (sample_dwave==True):
                #print("new_qpu_samples is HC (in conditional energy): {0}".format(0))
                sample_energies, sample_data = self._model.generate_samples_dwave(self._config.engine.n_valid_batch_size, energy, new_qpu_samples=0)
            else:
                sample_energies, sample_data = self._model.generate_samples(self._config.engine.n_valid_batch_size)
            sample_data = self._data_mgr.inv_transform(sample_data.detach().cpu().numpy())/1000. if self._config.data.scaled else sample_data.detach().cpu().numpy()
            conditioned_samples.append(torch.tensor(sample_data))
                        
        conditioned_samples = torch.cat(conditioned_samples, dim=0).numpy()
        self._hist_handler.update_samples(conditioned_samples)
            
    def _validate(self):
        logger.debug("engineCaloQV1::validate() : Running validation during a training epoch.")
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