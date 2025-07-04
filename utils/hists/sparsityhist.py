"""
Sparsity per layer histogram
"""

# Coffea histogramming library
from coffea import hist
import numpy as np

# Dataset labels
_LABELS = ["input", "recon", "samples", "samples_dwave"]

class SparsityHist(object):
    def __init__(self, start_idx, end_idx, min_bin=0, max_bin=1, n_bins=25):
        self._hist = hist.Hist(label="Events",
                               axes=(hist.Cat("dataset", "dataset type"),
                                     hist.Bin("sparsity", "Sparsity", n_bins, min_bin, max_bin)))
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._scale = "linear"
        self._data_dict = {key:[] for key in _LABELS}
        
    def update(self, in_data, recon_data, sample_data, sample_dwave_data):
        labels = ["input", "recon", "samples", "samples_dwave"]
        datasets = [in_data, recon_data, sample_data, sample_dwave_data]
        layer_datasets = [dataset[:, self._start_idx:self._end_idx] for dataset in datasets]
        layer_sparsities = [np.count_nonzero(layer_dataset, axis=1)/layer_dataset.shape[1] for layer_dataset in layer_datasets]
        
        for label, layer_sparsity in zip(labels, layer_sparsities):
            self._hist.fill(dataset=label, sparsity=layer_sparsity)
            self._data_dict[label].extend(layer_sparsity)
    
    def clear(self):
        self._hist.clear()
        self._data_dict = {key:[] for key in _LABELS}
        
    def get_hist(self):
        return self._hist
    
    def get_scale(self):
        return self._scale
    
    def get_data_dict(self):
        return self._data_dict