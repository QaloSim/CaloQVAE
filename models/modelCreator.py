"""
ModelCreator - Interface between run scripts and models.

Provides initialisation of models.

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import os
import torch
import wandb

from CaloQVAE import logging
logger = logging.getLogger(__name__)

#import defined models
from models.autoencoders.autoencoder import AutoEncoder
from models.autoencoders.sparseAE import SparseAutoEncoder
from models.autoencoders.variationalAE import VariationalAutoEncoder
from models.autoencoders.hierarchicalVAE import HierarchicalVAE
from models.autoencoders.conditionalVAE import ConditionalVariationalAutoEncoder
from models.autoencoders.sequentialVAE import SequentialVariationalAutoEncoder
from models.autoencoders.discreteVAE import DiVAE
from models.autoencoders.dvaepp import DiVAEPP
from models.autoencoders.gumbolt import GumBolt
from models.autoencoders.dvaeppcalo import DiVAEPPCalo
from models.autoencoders.gumboltCalo import GumBoltCalo
from models.autoencoders.gumboltCaloV2 import GumBoltCaloV2
from models.autoencoders.gumboltCaloV3 import GumBoltCaloV3
from models.autoencoders.gumboltCaloV4 import GumBoltCaloV4
from models.autoencoders.gumboltCaloV5 import GumBoltCaloV5
from models.autoencoders.gumboltCaloV6 import GumBoltCaloV6
from models.autoencoders.gumboltCaloV7 import GumBoltCaloV7
from models.autoencoders.gumboltCaloCRBM import GumBoltCaloCRBM
from models.autoencoders.gumboltCaloPRBM import GumBoltCaloPRBM
from models.autoencoders.atlasVAE import ATLASVAE

_MODEL_DICT={
    "AE": AutoEncoder, 
    "sparseAE": SparseAutoEncoder,
    "VAE": VariationalAutoEncoder,
    "cVAE": ConditionalVariationalAutoEncoder,
    "sVAE": SequentialVariationalAutoEncoder,
    "HiVAE": HierarchicalVAE,
    "DiVAE": DiVAE,
    "DiVAEpp": DiVAEPP,
    "gumBolt": GumBolt,
    "DiVAEppCalo": DiVAEPPCalo,
    "GumBoltCalo": GumBoltCalo,
    "GumBoltCaloV2": GumBoltCaloV2,
    "GumBoltCaloV3": GumBoltCaloV3,
    "GumBoltCaloV4": GumBoltCaloV4,
    "GumBoltCaloV5": GumBoltCaloV5,
    "GumBoltCaloV6": GumBoltCaloV6,
    "GumBoltCaloV7": GumBoltCaloV7,
    "GumBoltCaloCRBM": GumBoltCaloCRBM,
    "GumBoltCaloPRBM": GumBoltCaloPRBM,
    "ATLASVAE": ATLASVAE
}

class ModelCreator(object):

    def __init__(self, cfg=None):
        self._config=cfg

        self._model=None
        self._default_activation_fct=None
    
    def init_model(self, dataMgr=None):

        for key, model_class in _MODEL_DICT.items(): 
            if key.lower()==self._config.model.model_type.lower():
                logger.info("Initialising Model Type {0}".format(self._config.model.model_type))

                #TODO change init arguments. Ideally, the model does not carry
                #specific information about the dataset. 
                self.model = model_class(
                    flat_input_size=dataMgr.get_flat_input_size(),
                    train_ds_mean=dataMgr.get_train_dataset_mean(),
                    activation_fct=self._default_activation_fct,
                    cfg=self._config)
                
                return self.model
        logger.error("Unknown Model Type. Make sure your model is registered in modelCreator._MODEL_DICT.")
        raise NotImplementedError

    @property
    def model(self):
        assert self._model is not None, "Model is not defined."
        return self._model

    @model.setter
    def model(self,model):
        self._model=model

    @property
    def default_activation_fct(self):
        return self._default_activation_fct

    @default_activation_fct.setter
    def default_activation_fct(self, act_fct):
        self._default_activation_fct=act_fct
    
    def save_state(self, cfg_string='test'):
        logger.info("Saving state")
        path = os.path.join(wandb.run.dir, "{0}.pth".format(cfg_string))
        
        # Extract modules from the model dict and add to start_dict 
        modules=list(self._model._modules.keys())
        state_dict={module: getattr(self._model, module).state_dict() for module in modules}
        
        # Save the model parameter dict
        torch.save(state_dict, path)
        
    def load_state(self, run_path, device):
        logger.info("Loading state")
        model_loc = run_path
        
        # Open a file in read-binary mode
        with open(model_loc, 'rb') as f:
            # Interpret the file using torch.load()
            checkpoint=torch.load(f, map_location=device)

            logger.info("Loading weights from file : {0}".format(run_path))
            
            local_module_keys=list(self._model._modules.keys())
            for module in checkpoint.keys():
                if module in local_module_keys:
                    print("Loading weights for module = ", module)
                    getattr(self._model, module).load_state_dict(checkpoint[module])

if __name__=="__main__":
    logger.info("Willkommen!")
    mm=ModelCreator()
    logger.info("Success!")
