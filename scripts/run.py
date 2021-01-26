# -*- coding: utf-8 -*-
"""
Main runscript

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

#external libraries
import os,sys
import pickle
import datetime
import gif

import torch
torch.manual_seed(1)
import numpy as np
import matplotlib.pyplot as plt

#self defined imports
from DiVAE import logging
logger = logging.getLogger(__name__)
from DiVAE import config

from data.dataManager import DataManager

from utils.plotProvider import PlotProvider

def run(modelMaker=None):

    #container for our Dataloaders
    dataMgr=DataManager()
    #initialise data loaders
    dataMgr.init_dataLoaders()
    #run pre processing: get/set input dimensions and mean of train dataset
    dataMgr.pre_processing()
    #add dataMgr instance to modelMaker namespace
    modelMaker.register_dataManager(dataMgr)

    #set parameters relevant for this run
    date=datetime.datetime.now().strftime("%y%m%d")

    config_string="_".join(str(i) for i in [config.model_type,
                                        config.data_type,
                                        config.n_train_samples,
                                        config.n_test_samples,
                                        config.n_batch_samples,
                                        config.n_epochs,
                                        config.learning_rate,
                                        config.n_latent_hierarchy_lvls,
                                        config.n_latent_nodes,
                                        config.activation_fct,
                                        config.tag])
    
    if config.data_type=='calo': 
        config_string+="_nlayers_{0}_{1}".format(len(config.calo_layers),config.particle_type)

    if config.activation_fct.lower()=="relu":
        modelMaker.default_activation_fct=torch.nn.ReLU() 
    elif config.activation_fct.lower()=="tanh":
        modelMaker.default_activation_fct=torch.nn.ReLU() 
    else:
        logger.warning("Setting identity as default activation fct")
        modelMaker.default_activation_fct=torch.nn.Identity() 
    
    #instantiate the chosen model
    model=modelMaker.init_model()
    #create the NN infrastructure
    model.create_networks()
    model.print_model_info()

    #instantiate and register optimisation algorithm
    modelMaker.optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    #if we load a model from a file, we don't need to train
    if config.load_model:
        #TODO needs re-implementation
        # modelMaker.load_model(set_eval=True)
        #        if config.load_model:
        #   config_string=config.input_model.split("/")[-1].replace('.pt','')
 
        pass
    else:
        for epoch in range(1, config.n_epochs+1):   
            train_loss = modelMaker.fit(epoch=epoch, is_training=True)
            test_loss = modelMaker.fit(epoch=epoch, is_training=False)
    
    #TODO improve the save functionality
    if config.save_model:
        modelMaker.save_model(config_string)
        if model.type=="DiVAE": 
            modelMaker.save_rbm(config_string)

    #sample generation
    if config.generate_samples:
        output_generated=modelMaker.generate_samples()

    if config.create_plots:
        #call a forward method derivative - for output object.
        eval_output=modelMaker.evaluate()
        #create plotting infrastructure
        pp=PlotProvider(config_string=config_string,date_tag=date)
        #TODO is there a neater integration than to add this as member?
        pp.data_dimensions=dataMgr.get_input_dimensions()
        #create plot
        pp.plot(eval_output)
    logger.info("run() finished successfully.")

if __name__=="__main__":
    logger.info("Starting main executable.")

    #check if output path exists, create if necessary
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    
    #create model handling object
    from utils.modelMaker import ModelMaker
    modelMaker=ModelMaker()

    #run the ting
    run(modelMaker)

    logger.info("Auf Wiedersehen!")
