"""
Sample energy histogram
"""

# Coffea histogramming library
from coffea import hist
import numpy as np

# Dataset labels
_LABELS = ["true", "samples"]

class SampleEnergyHist(object):
    def __init__(self, min_bin=0, max_bin=100, n_bins=100):
        self._hist = hist.Hist(label="Events",
                               axes=(hist.Cat("dataset", "dataset type"),
                                     hist.Bin("E", "Observed Energy (GeV)",
                                              n_bins, min_bin, max_bin)))
        self._scale = "linear"
        
    def update(self, sample_data):
        datasets = [sample_data]
        datasets = [data.sum(axis=1) for data in datasets]
        
        for label, dataset in zip(_LABELS, datasets):
            self._hist.fill(dataset=label, E=dataset)
            
    def clear(self):
        self._hist.clear()
        
    def get_hist(self):
        return self._hist
    
    def get_scale(self):
        return self._scale