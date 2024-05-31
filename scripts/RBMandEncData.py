import hydra
from hydra.utils import instantiate
from hydra import initialize, compose

import sys
import os
import getpass

os.chdir('/home/' + getpass.getuser() + '/Projects/CaloQVAE/')
sys.path.insert(1, '/home/' + getpass.getuser() + '/Projects/CaloQVAE/')

#external libraries
import os
import pickle
import datetime
import sys
import yaml
import json

import torch.optim as optim
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
torch.manual_seed(32)
import numpy as np
np.random.seed(32)
import matplotlib.pyplot as plt
import hydra
from hydra.utils import instantiate

from omegaconf import OmegaConf

import time

# PyTorch imports
from torch import device, load, save
from torch.nn import DataParallel
from torch.cuda import is_available

# Add the path to the parent directory to augment search for module
sys.path.append(os.getcwd())
    
# Weights and Biases
import wandb

#self defined imports
from CaloQVAE import logging

logger = logging.getLogger(__name__)

from data.dataManager import DataManager
from utils.plotting.plotProvider import PlotProvider
from engine.engine import Engine
from models.modelCreator import ModelCreator

from utils.plotting.HighLevelFeatures import HighLevelFeatures as HLF
HLF_1_photons = HLF('photon', filename='/raid/javier/Datasets/CaloVAE/data/atlas/binning_dataset_1_photons.xml', wandb=False)
HLF_1_pions = HLF('pion', filename='/raid/javier/Datasets/CaloVAE/data/atlas/binning_dataset_1_pions.xml', wandb=False)
HLF_1_electron = HLF('electron', filename='/raid/javier/Datasets/CaloVAE/data/atlas_dataset2and3/binning_dataset_2.xml', wandb=False)

hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(version_base=None, config_path="../configs")

###############################



# # divine-valentine-309 | CNN + cond + scaled data + dec charm
# run_path = "/home/javier/Projects/CaloQVAE/outputs/2024-02-15/17-36-19/wandb/run-20240215_173620-l3i43zja/files/GumBoltAtlasPRBMCNN_atlas_default_latest.pth"
# modelname = 'divine-valentine-309'


# # # toasty-cherry-310 | CNN + cond + scaled data + hits dec uncond
# # run_path = "/home/javier/Projects/CaloQVAE/outputs/2024-02-15/22-10-50/wandb/run-20240215_221050-caevb6ld/files/GumBoltAtlasPRBMCNN_atlas_default_best.pth"
# # modelname = 'toasty-cherry-310'

# # robust-tree-339 | CNN + cond + scaled data
# run_path = "/home/javier/Projects/CaloQVAE/outputs/2024-02-27/18-38-25/wandb/run-20240301_174432-y5uczif5/files/GumBoltAtlasPRBMCNN_atlas_default_best.pth"
# modelname = 'robust-tree-339'

# morning-breeze-420 | CNN + cond + scaled data + Cyl EncDec + lin/sqrt/log LONG energy encoded + CRBM 1st Partition Binv2
# run_path = "/home/javier/Projects/CaloQVAE/outputs/2024-05-18/15-22-04/wandb/run-20240518_152205-pi1sujcx/files/AtlasConditionalQVAE_atlas_default_150.pth"
run_path = "/home/javier/Projects/CaloQVAE/outputs/2024-05-18/15-22-04/wandb/run-20240524_194939-pi1sujcx/files/AtlasConditionalQVAE_atlas_default_200.pth"
modelname = 'morning-breeze-420'
     
datascaled = 'reduced'
R = 0.01
reducedata = False
scaled=True 
sample_size = 1024
######################################

def main():
    config=compose(config_name="config.yaml")
    wandb.init(project="caloqvae", entity="qvae", config=config, mode='disabled')
    modelCreator = ModelCreator(cfg=config)
    dataMgr = DataManager(cfg=config)
    #initialise data loaders
    dataMgr.init_dataLoaders()
    #run pre processing: get/set input dimensions and mean of train dataset
    dataMgr.pre_processing()

    if config.model.activation_fct.lower()=="relu":
        modelCreator.default_activation_fct=torch.nn.ReLU()
    elif config.model.activation_fct.lower()=="tanh":
        modelCreator.default_activation_fct=torch.nn.Tanh()
    else:
        logger.warning("Setting identity as default activation fct")
        modelCreator.default_activation_fct=torch.nn.Identity()

    #instantiate the chosen model
    #loads from file 
    model=modelCreator.init_model(dataMgr=dataMgr)

    #create the NN infrastructure
    model.create_networks()
    
    #Not printing much useful info at the moment to avoid clutter. TODO optimise
    model.print_model_info()
    dev = "cuda:{0}".format(config.gpu_list[0])

    # Send the model to the selected device
    # model.to(dev)
    # Log metrics with wandb
    wandb.watch(model)

    # For some reason, need to use postional parameter cfg instead of named parameter
    # with updated Hydra - used to work with named param but now is cfg=None 
    engine=instantiate(config.engine, config)

    #TODO for some reason hydra double instantiates the engine in a
    #newer version if cfg=config is passed as an argument. This is a workaround.
    #Find out why that is...
    engine._config=config
    #add dataMgr instance to engine namespace
    engine.data_mgr=dataMgr
    #add device instance to engine namespace
    engine.device=dev    
    #instantiate and register optimisation algorithm
    engine.optimiser = torch.optim.Adam(model.parameters(),
                                        lr=config.engine.learning_rate)
    #add the model instance to the engine namespace
    engine.model = model
    # add the modelCreator instance to engine namespace
    engine.model_creator = modelCreator
    engine.model = engine.model.to(dev)
    
    train_loader,test_loader,val_loader = engine.data_mgr.create_dataLoader()
    
    # load_state(model, run_path, 'cuda:{0}'.format(cfg.gpu_list[0]))
    # load_state(model, run_path, dev)
    modelCreator.load_state(run_path, dev)
    engine.model.eval();
    
    logger.info(f'Generating RBM sample from Gibss')
    energy_rbm_data = get_rbm_hist(engine, config, val_loader)
    logger.info(f'Done...')
    logger.info(f'Generating multiple samples for single event via multiple encodings.')
    draw_hist_per_energy(engine, dev, config, val_loader, modelname, energy_rbm_data, range_len=5000)
    logger.info(f'Done...')
    logger.info(f'Generating samples plots for incidence energy tuning in decoder')
    draw_sample(engine, dev, config, val_loader, modelname)
    # logger.info(f'Computing distances between true incidence energy label and incidence energy which minimizes MSE and sparsity.')
    # get_diff_btw_true_and_argmin_einc(engine, dev, config, val_loader, modelname, range_len=5000)
    logger.info(f'Finished!')


def get_rbm_hist(engine, config, val_loader):
    
    partition_size=config.model.n_latent_nodes

    energy_encoded_data = []

    engine.model.eval()

    with torch.no_grad():
        for xx in val_loader:
            in_data, true_energy, in_data_flat = engine._preprocess(xx[0],xx[1])
            if reducedata:
                in_data = engine._reduce(in_data, true_energy, R=R)
            # enIn = torch.cat((in_data, true_energy), dim=1)
            # beta, post_logits, post_samples = engine.model.encoder(enIn, False)
            beta, post_logits, post_samples = engine.model.encoder(in_data, true_energy, False)
            post_samples = torch.cat(post_samples, 1)
            post_samples_energy = engine.model.stater.energy_samples(post_samples[:,0:partition_size], post_samples[:,partition_size:2*partition_size], 
                                                     post_samples[:,2*partition_size:3*partition_size], post_samples[:,3*partition_size:4*partition_size], 1.0 )

            energy_encoded_data.append(post_samples_energy.detach().cpu())

    energy_encoded_data = torch.cat(energy_encoded_data, dim=0)

    p1,p2,p3,p4 = post_samples[:,0:partition_size], post_samples[:,partition_size:2*partition_size], \
                                                     post_samples[:,2*partition_size:3*partition_size], post_samples[:,3*partition_size:4*partition_size]

    energy_rbm_data = []
    with torch.no_grad():
        for i in range(10):
            # if i == 0:
                # p1, p2, p3, p4 = engine.model.stater.block_gibbs_sampling_ais(1.0)
            # else:
                # p1, p2, p3, p4 = engine.model.stater.block_gibbs_sampling_ais(1.0, p1, p2, p3, p4)
            p1, p2, p3, p4 = engine.model.sampler.block_gibbs_sampling()
            rbm_samples_energy = engine.model.stater.energy_samples(p1, p2, p3, p4, 1.0)
            energy_rbm_data.append(rbm_samples_energy.detach().cpu())

    energy_rbm_data = torch.cat(energy_rbm_data, dim=0)
    return energy_rbm_data
    
    
def draw_hist_per_energy(engine, dev, config, val_loader, modelname, energy_rbm_data, range_len=106):
    
    partition_size = config.model.n_latent_nodes
    cmap = plt.cm.viridis

    mean_rbm_energy = np.array([[0,0,0]])

    plt.figure(figsize=(8,6))
    for ind in range(0,range_len):
        en_ind = torch.Tensor([[val_loader.__dict__['dataset'].__dict__['_true_energies'].reshape(-1)[ind]]]).repeat(sample_size, 1)
        x_ind = val_loader.__dict__['dataset'].__dict__['_images']['showers'].__dict__['_image'][ind].unsqueeze(0).repeat(sample_size, 1)

        energy_encoded_data_per_energy = []
        for i in range(5):
            beta, post_logits, post_samples = engine.model.encoder(x_ind.to(dev), en_ind.to(dev), False)
            post_samples = torch.cat(post_samples, 1)
            post_samples_energy = engine.model.stater.energy_samples(post_samples[:,0:partition_size], post_samples[:,partition_size:2*partition_size], 
                                                             post_samples[:,2*partition_size:3*partition_size], post_samples[:,3*partition_size:4*partition_size], 1.0 )

            energy_encoded_data_per_energy.append(post_samples_energy.detach().cpu())
        energy_encoded_data_per_energy = torch.cat(energy_encoded_data_per_energy, dim=0)

        # mean_rbm_energy = np.concatenate((mean_rbm_energy, np.array([[energy_encoded_data_per_energy.mean(), en_ind[0].item()]])), axis=0)
        mean_rbm_energy = np.concatenate((mean_rbm_energy, np.array([[energy_encoded_data_per_energy.mean(), en_ind[0].item(), energy_encoded_data_per_energy.std()]])), axis=0)
        
        if ind<=106:
            lbl = np.round(en_ind[0].item()/1000,2)
            plt.hist(energy_encoded_data_per_energy.numpy(), color=cmap(lbl/1000), bins=60, linewidth=2.5, density=True, label=f'{lbl} GeV', log=True, alpha=0.5) 

    plt.hist(energy_rbm_data.numpy(), bins=20, color="orange", density=True, fc=(1, 0, 1, 0.5), log=True, histtype='step', linewidth=2.5)
    # plt.legend()
    plt.xlabel("RBM Energy", fontsize=15)
    plt.ylabel("PDF", fontsize=15)
    plt.grid("True")
    plt.savefig(f'/home/javier/Projects/CaloQVAE/figs/{modelname}/RBM_energy_per_inc_energy_{modelname}.png')
    # plt.show()
    
    plt.figure(figsize=(8,6))
    plt.errorbar(mean_rbm_energy[1:,1]/1000, mean_rbm_energy[1:,0], yerr=mean_rbm_energy[1:,2], fmt='o', ecolor='lightgray', elinewidth=3, capsize=0, alpha=0.6, color="blue")
    # plt.scatter(mean_rbm_energy[1:,1]/1000, mean_rbm_energy[1:,0], marker='o', alpha=1, color="blue")
    plt.axhline(energy_rbm_data.numpy().std() + energy_rbm_data.numpy().mean(), label="RBM mean energy +/- std", linestyle="dashdot", color='orange')
    plt.axhline(-energy_rbm_data.numpy().std() + energy_rbm_data.numpy().mean(), linestyle="dashdot", color='orange')
    plt.grid(True)
    plt.xlabel("Incidence Energy (GeV)", fontsize=15)
    plt.ylabel("mean RBM energy", fontsize=15)
    plt.legend(framealpha=0.5, fontsize=15)
    plt.savefig(f'/home/javier/Projects/CaloQVAE/figs/{modelname}/RBM_energy_per_inc_energy_mean_{modelname}.png')
    # plt.show()
    
    
def error_btwn_input_and_recon_with_different_label(engine, dev, config, val_loader, modelname, ind, start=3, stop=6, num=50):
    # ind = 2

    en_ind = torch.Tensor([[val_loader.__dict__['dataset'].__dict__['_true_energies'].reshape(-1)[ind]]]).repeat(sample_size, 1)
    x_ind = val_loader.__dict__['dataset'].__dict__['_images']['showers'].__dict__['_image'][ind].unsqueeze(0).repeat(sample_size, 1)
    in_data = torch.tensor(engine._data_mgr.inv_transform(x_ind.detach().cpu().numpy()))

    log_space_values = np.logspace(start=start, stop=stop, num=num, base=10)

    with torch.no_grad():
        beta, post_logits, post_samples = engine.model.encoder(x_ind.to(dev), en_ind.to(dev), False)
        post_samples = torch.cat(post_samples, 1)

        er_m_list = []
        er_std_list = []
        
        sp_er_m_list = []
        sp_er_std_list = []
        for en_prime in log_space_values:
            en_ind_prime = torch.Tensor([[en_prime]]).repeat(sample_size, 1)
            output_hits, output_activations = engine.model.decoder(post_samples, en_ind_prime.to(dev))
            beta = torch.tensor(engine.model._config.model.output_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)

            output_activations = engine.model._inference_energy_activation_fct(output_activations) * engine.model._hit_smoothing_dist_mod(output_hits, beta, True)

            er = torch.pow(output_activations.sum(dim=1) - x_ind.to(dev).sum(dim=1), 2) #.cpu().numpy()
            er_m_list.append(er.mean().item()/1000)
            er_std_list.append(er.std().item()/1000)
            
            # spar_er = (((x_ind.to(dev).sign()==0).sum(dim=1)/x_ind.to(dev).shape[1] - (output_activations.sign()==0).sum(dim=1)/output_activations.shape[1])/((x_ind.to(dev).sign()==0).sum(dim=1)/x_ind.to(dev).shape[1] + 1)).cpu().numpy()
            # spar_er = (((x_ind.to(dev).sign()==0).sum(dim=1) / (output_activations.sign()==0).sum(dim=1))).cpu().numpy()
            spar_er = ((x_ind.to(dev).sign() - output_activations.sign()).abs().sum(dim=1)) #.cpu().numpy()
            sp_er_m_list.append(spar_er.mean().item())
            sp_er_std_list.append(spar_er.std().item())
            
    return er_m_list, er_std_list, log_space_values, en_ind[0].item(), in_data.sum(dim=1)[0].item(), sp_er_m_list, sp_er_std_list


def draw_sample(engine, dev, config, val_loader, modelname):
    er_m_list, er_std_list, log_space_values, en_inc, shower_sum_in, sp_er_m_list, sp_er_std_list = error_btwn_input_and_recon_with_different_label(engine, dev, config, val_loader, modelname, 0, num=50)

    plt.figure(figsize=(8,6))
    plt.errorbar(log_space_values, er_m_list, yerr=er_std_list, fmt='o', ecolor='gray', elinewidth=3, capsize=0, alpha=0.6, color="blue")
    plt.xscale("log")
    plt.yscale("log")
    plt.axvline(en_inc, label="Inc energy as input to encoder", linestyle="dashdot", color='black')
    # plt.axvline(shower_sum_in, c='r')
    plt.grid(True)
    plt.xlabel("Incidence Energy as input to decoder (GeV)", fontsize=15)
    plt.ylabel("MSE(input, prediction)", fontsize=15)
    plt.legend(framealpha=0.5, fontsize=15)
    plt.savefig(f'/home/javier/Projects/CaloQVAE/figs/{modelname}/MSE_energy_per_inc_energy_mean_{modelname}.png')
    # plt.show()
    
    plt.figure(figsize=(8,6))
    plt.errorbar(log_space_values, sp_er_m_list, yerr=sp_er_std_list, fmt='o', ecolor='lightgray', elinewidth=3, capsize=0, alpha=0.6, color='blue')
    plt.xscale("log")
    plt.yscale("log")
    plt.axvline(en_inc, label="Inc energy as input to encoder", linestyle="dashdot", color='black')
    # plt.axvline(shower_sum_in, c='r')
    plt.grid(True)
    plt.xlabel("Incidence Energy as input to decoder (GeV)", fontsize=15)
    plt.ylabel("sparsity(input, prediction)", fontsize=15)
    plt.legend(framealpha=0.5, fontsize=15)
    plt.savefig(f'/home/javier/Projects/CaloQVAE/figs/{modelname}/sparsity_energy_per_inc_energy_mean_{modelname}.png')
    # plt.show()
    
    
def get_diff_btw_true_and_argmin_einc(engine, dev, config, val_loader, modelname, range_len=106):
    # res_ar = torch.tensor([[0,0,0,0]])
# res_ar = np.array([[0,0,0,0]])

    res_ar = torch.zeros(range_len,4)

    for i in range(range_len):
        logger.info(f'{i}')
        er_m_list, er_std_list, log_space_values, en_inc, shower_sum_in, sp_er_m_list, sp_er_std_list = error_btwn_input_and_recon_with_different_label(engine, dev, config, val_loader, modelname, i, num=200)

        energy_label_min = log_space_values[torch.tensor(er_m_list).argmin().item()]
        energy_label_min_spar = log_space_values[torch.tensor(sp_er_m_list).argmin().item()]
        # res_ar = np.concatenate((res_ar, np.array([[energy_label_min, en_inc, shower_sum_in, energy_label_min_spar]])), axis=0)
        # res_ar = torch.cat([res_ar, torch.tensor([[energy_label_min, en_inc, shower_sum_in, energy_label_min_spar]])], dim=0)
        res_ar[i,:] = torch.tensor([[energy_label_min, en_inc, shower_sum_in, energy_label_min_spar]])
        
    # plt.scatter(res_ar[1:,1]/1000, (res_ar[1:,0] - res_ar[1:,1])/1000, color='b')
    # plt.grid(True)
    # plt.xlabel("Incidence Energy as input to decoder (GeV)", fontsize=15)
    # plt.ylabel("| true Eᵢ - argmin(MSE(true Eᵢ, Eᵢ'))", fontsize=15)
    # # plt.legend(framealpha=0.5, fontsize=15)
    # plt.show()
    
    plt.figure(figsize=(8,6))
    y = (res_ar[1:,3] - res_ar[1:,1])/1000
    x = res_ar[1:,1]/1000
    x_pos = x[y>0]
    y_pos = y[y>0]
    x_neg = x[y<0]
    y_neg = y[y<0]
    # plt.scatter(res_ar[1:,1]/1000, abs(res_ar[1:,3] - res_ar[1:,1])/1000, color='b')
    # plt.scatter(res_ar[1:,1]/1000, (res_ar[1:,3] - res_ar[1:,1])/1000, color='b')
    plt.scatter(x_pos, abs(y_pos), color='b', label = 'y positive')
    plt.scatter(res_ar[1:,1]/1000, (res_ar[1:,0] - res_ar[1:,1])/1000, color='red', alpha=0.5)
    plt.grid(True)
    plt.xlabel("Incidence Energy as input to decoder (GeV)", fontsize=15)
    plt.ylabel("| Eᵢ - Eᵢ'|", fontsize=15)
    plt.legend(['sparsity', 'shower'], framealpha=0.5, fontsize=15)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.savefig(f'/home/javier/Projects/CaloQVAE/figs/{modelname}/diff_energy_per_inc_energy_mean_{modelname}.png')
    # plt.show()
    
    
    
    
    
if __name__=="__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Hasta pronto!")