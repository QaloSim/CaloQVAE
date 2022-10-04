"""
Base Class of Engines. Defines properties and methods.
"""

import torch

# Weights and Biases
import wandb

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class EngineBase(object):

    def __init__(self, cfg=None, **kwargs):
        super(EngineBase,self).__init__()

        self._config = cfg
        
        self._model = None
        self._optimiser = None
        self._data_mgr = None
        self._device = None
        self._model_creator = None

    @property
    def model(self):
        return self._model
    
    @model.setter   
    def model(self,model):
        self._model=model

    @property
    def optimiser(self):
        return self._optimiser
    
    @optimiser.setter   
    def optimiser(self,optimiser):
        self._optimiser=optimiser
    
    @property
    def data_mgr(self):
        return self._data_mgr
    
    @data_mgr.setter   
    def data_mgr(self,data_mgr):
        assert data_mgr is not None, "Empty Data Manager"
        self._data_mgr=data_mgr
        
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        self._device=device
        
    @property
    def model_creator(self):
        return self._model_creator
    
    @model_creator.setter
    def model_creator(self, model_creator):
        assert model_creator is not None
        self._model_creator = model_creator
    
    def generate_samples(self):
        raise NotImplementedError

    def fit(self, epoch, is_training=True):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError