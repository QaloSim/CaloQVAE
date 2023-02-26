#!/usr/bin/python3
"""
Main executable. The run() method steers data loading, model creation, training
and evaluation by calling the respective interfaces.

Author: Abhishek <abhishek@myumanitoba.ca>
Author: Eric Drechsler <eric.drechsler@cern.ch>
"""

#external libraries
import os
import pickle
import datetime
import sys

import torch
torch.manual_seed(32)
import numpy as np
np.random.seed(32)
import matplotlib.pyplot as plt
import hydra
from hydra.utils import instantiate

from omegaconf import OmegaConf

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

@hydra.main(config_path="../configs", config_name="config")

def main(cfg=None):
    # initialise wandb logging. Note that this function has many more options,
    # reference: https://docs.wandb.ai/ref/python/init
    # this is the setting for individual, ungrouped runs
    # Use mode='disabled' to prevent logging
    mode = 'online' if cfg.wandb_enabled else 'disabled'
    wandb.init(project="caloqvae", entity="qvae", config=cfg, mode=mode)
    # run the ting
    run(config=cfg)

def run(config=None):

    """
    Run m
    """
    #create model handling object
    modelCreator = ModelCreator(cfg=config)

    #container for our Dataloaders
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

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    # Load the model on the GPU if applicable
    dev = None
    if (config.device == 'gpu') and config.gpu_list:
        logger.info('Requesting GPUs. GPU list :' + str(config.gpu_list))
        devids = ["cuda:{0}".format(x) for x in list(config.gpu_list)]
        logger.info("Main GPU : " + devids[0])
        
        
        if is_available():
            print(devids[0])
            dev = device(devids[0])
            if len(devids) > 1:
                logger.info(f"Using DataParallel on {devids}")
                model = DataParallel(model, device_ids=list(config.gpu_list))
            logger.info("CUDA available")
        else:
            dev = device('cpu')
            logger.info("CUDA unavailable")
    else:
        logger.info('Requested CPU or unable to use GPU. Setting CPU as device.')
        dev = device('cpu')
        
    # Send the model to the selected device
    model.to(dev)
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

    if config.load_state:
        assert config.run_path != 0
        config_string = "_".join(str(i) for i in [config.model.model_type, config.data.data_type, config.tag])
        modelCreator.load_state(config.run_path, dev)
    try:
        n_iter = config.engine.n_epochs
    except:
        n_iter = config.engine.n_batches
    for epoch in range(1, n_iter+1):
        if "train" in config.task:
            engine.fit(epoch=epoch, is_training=True, mode="train")

        if "validate" in config.task:
            engine.fit(epoch=epoch, is_training=False, mode="validate")

        if "sample" in config.task:
            engine.fit(epoch=epoch, is_training=False)

    if "test" in config.task:
        engine.fit(epoch=epoch, is_training=False, mode="test")


    if config.save_state:
        config_string = "_".join(str(i) for i in [config.model.model_type, 
                                                  config.data.data_type,
                                                  config.tag, "latest"])
        modelCreator.save_state(config_string)

    logger.info("run() finished successfully.")


if __name__=="__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Auf Wiedersehen!")
