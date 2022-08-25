"""
PyTorch implementation of a quadripartite Boltzmann machine with a Pegasus/Advantage QPU topology
"""
import numpy as np
import torch
from torch import nn

from DiVAE import logging
logger = logging.getLogger(__name__)

class PegasusRBM(nn.Module):
    def __init__(self):
        
    

if __name__=="__main__":
    logger.debug("Testing chimeraRBM")
    cRBM = PegasusRBM(8, 8)
    print(cRBM.weights)
    logger.debug("Success")