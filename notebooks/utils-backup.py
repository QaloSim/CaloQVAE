# Helper functions for notebooks
import torch
import numpy as np
import matplotlib.pyplot as plt

def sample_energies(rbm, rbm_vis, rbm_hid):
    """
    Compute the energies of samples produced by the RBM

    Returns:
        rbm_energy_exp : -vis^T W hid - a^T hid - b^T vis
    """
    # Broadcast W to (pcd_batchSize * nVis * nHid)
    w, vbias, hbias = rbm.weights, rbm.visible_bias, rbm.hidden_bias
    w = w + torch.zeros((rbm_vis.size(0),) + w.size(), device=rbm_vis.device)
    vbias = vbias.to(rbm_vis.device)
    hbias = hbias.to(rbm_hid.device)

    # Prepare H, V for torch.matmul()
    # Change V.size() from (batchSize * nVis) to (batchSize * 1 * nVis)
    vis = rbm_vis.unsqueeze(2).permute(0, 2, 1)
    # Change H.size() from (batchSize * nHid) to (batchSize * nHid * 1)
    hid = rbm_hid.unsqueeze(2)

    batch_energies = (- torch.matmul(vis, torch.matmul(w, hid)).reshape(-1) 
                      - torch.matmul(rbm_vis, vbias)
                      - torch.matmul(rbm_hid, hbias))

    return batch_energies

def sample_energies_exp(rbm_weights, rbm_visible_bias, rbm_hidden_bias, rbm_vis, rbm_hid):
    """
    Compute the energies of samples produced by the RBM

    Returns:
        rbm_energy_exp : -vis^T W hid - a^T hid - b^T vis
    """
    # Broadcast W to (pcd_batchSize * nVis * nHid)
    w, vbias, hbias = rbm_weights, rbm_visible_bias, rbm_hidden_bias
    w = w + torch.zeros((rbm_vis.size(0),) + w.size(), device=rbm_vis.device)
    vbias = vbias.to(rbm_vis.device)
    hbias = hbias.to(rbm_hid.device)

    # Prepare H, V for torch.matmul()
    # Change V.size() from (batchSize * nVis) to (batchSize * 1 * nVis)
    vis = rbm_vis.unsqueeze(2).permute(0, 2, 1)
    # Change H.size() from (batchSize * nHid) to (batchSize * nHid * 1)
    hid = rbm_hid.unsqueeze(2)

    batch_energies = (- torch.matmul(vis, torch.matmul(w, hid)).reshape(-1) 
                      - torch.matmul(rbm_vis, vbias)
                      - torch.matmul(rbm_hid, hbias))
    return batch_energies

def plot_sample_energies(energies):
    """
    Plot the energies of the samples produced by the histograms        
    """
    fig, ax = plt.subplots(figsize=(40, 16))
    
    ax.hist(energies, bins=100)
    
    ax.set_xlabel("Energy", fontsize=60)
    ax.set_ylabel("Frequency", fontsize=60)
    
    ax.tick_params(axis='both', which='major', labelsize=60)
    
    plt.show()
    plt.close()
    
def batch_dwave_samples(response):
    """
    sampler.sample_ising() method returns a nested SampleSet structure
    with unique samples, energies and number of occurences stored in dict 
    
    Extract those values and construct a batch_size * (num_vis+num_hid) numpy array
    
    Returns:
        batch_samples : batch_size * (num_vis+num_hid) numpy array of samples collected by the DWave sampler
        batch_energies : batch_size * 1 numpy array of energies of samples
    """
    samples = []
    energies = []
    
    for sample_info in response.data():
        uniq_sample = list(sample_info[0].values())
        sample_energy = sample_info[1]
        num_occurences = sample_info[2]
        
        samples.extend([uniq_sample]*num_occurences)
        energies.extend([sample_energy]*num_occurences)
        
    batch_samples = np.array(samples)
    batch_energies = np.array(energies).reshape(-1)
        
    return batch_samples, batch_energies

def plot_betas(betas):
    """
    Plot the estimates of beta during the beta estimation procedure   
    """
    fig, ax = plt.subplots(figsize=(40, 16))
    
    plt.plot(np.arange(len(betas)), betas)
    
    ax.set_xlabel("Iteration", fontsize=60)
    ax.set_ylabel("Beta", fontsize=60)
    
    ax.tick_params(axis='both', which='major', labelsize=60)
    
    plt.show()
    plt.close()