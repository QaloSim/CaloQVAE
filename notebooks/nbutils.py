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

def sample_energies_qpu(rbm_weights, rbm_visible_bias, rbm_hidden_bias, rbm_vis, rbm_hid):
    """
    Compute the energies of samples produced by the RBM

    Returns:
        rbm_energy_exp : vis^T W hid + a^T hid + b^T vis
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

    batch_energies = (torch.matmul(vis, torch.matmul(w, hid)).reshape(-1) 
                      + torch.matmul(rbm_vis, vbias)
                      + torch.matmul(rbm_hid, hbias))
    return batch_energies

def rbm_to_ising(rbm_weights, rbm_visible_bias, rbm_hidden_bias):
    """
    Transform the parameters of an RBM into the parameters of an Ising model
    """
    ising_weights = rbm_weights/4.
    ising_vbias = rbm_visible_bias/2. + torch.sum(rbm_weights, dim=1)/4.
    ising_hbias = rbm_hidden_bias/2. + torch.sum(rbm_weights, dim=0)/4.
    
    return ising_weights, ising_vbias, ising_hbias
    

def sample_energies_exp_ising(rbm_weights, rbm_visible_bias, rbm_hidden_bias, rbm_vis, rbm_hid):
    """
    Compute the energies of samples produced by the RBM transformed into Ising samples

    Returns:
        rbm_energy_exp : -vis^T W hid - a^T hid - b^T vis + K
    """
    # Broadcast W to (pcd_batchSize * nVis * nHid)
    w, vbias, hbias = rbm_weights, rbm_visible_bias, rbm_hidden_bias
    
    w_sum = torch.sum(torch.sum(w, dim=0))
    vbias_sum = torch.sum(vbias, dim=0)
    hbias_sum = torch.sum(hbias, dim=0)
    
    print(w_sum, vbias_sum, hbias_sum)
    
    ising_offset = (w_sum/2. + vbias_sum + hbias_sum)/2.
    
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
    batch_energies_shifted = batch_energies - ising_offset
    return batch_energies_shifted

def ising_energies_exp(ising_weights, ising_visible_bias, ising_hidden_bias, ising_vis, ising_hid):
    """
    Compute the energies of samples produced by an Ising model

    Returns:
        ising_energies : + vis^T W hid + a^T hid + b^T vis
    """
    # Broadcast W to (pcd_batchSize * nVis * nHid)
    w, vbias, hbias = ising_weights, ising_visible_bias, ising_hidden_bias
    w = w + torch.zeros((ising_vis.size(0),) + w.size(), device=ising_vis.device)
    vbias = vbias.to(ising_vis.device)
    hbias = hbias.to(ising_hid.device)

    # Prepare H, V for torch.matmul()
    # Change V.size() from (batchSize * nVis) to (batchSize * 1 * nVis)
    vis = ising_vis.unsqueeze(2).permute(0, 2, 1)
    # Change H.size() from (batchSize * nHid) to (batchSize * nHid * 1)
    hid = ising_hid.unsqueeze(2)

    ising_energies = (torch.matmul(vis, torch.matmul(w, hid)).reshape(-1) 
                      + torch.matmul(ising_vis, vbias)
                      + torch.matmul(ising_hid, hbias))
    return ising_energies

def plot_sample_energies(energies):
    """
    Plot the energies of the samples produced by the histograms        
    """
    fig, ax = plt.subplots(figsize=(40, 16))
    
    ax.hist(energies, bins=100)
    
    ax.set_xlabel("Energy", fontsize=60)
    ax.set_ylabel("Frequency", fontsize=60)
    
    ax.tick_params(axis='both', which='major', labelsize=60)
    ax.grid(True)
    
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
    
def load_state(model, run_path, device):
    model_loc = run_path
    
    # Open a file in read-binary mode
    with open(model_loc, 'rb') as f:
        # Interpret the file using torch.load()
        checkpoint=torch.load(f, map_location=device)
            
        local_module_keys=list(model._modules.keys())
        for module in checkpoint.keys():
            if module in local_module_keys:
                print("Loading weights for module = ", module)
                getattr(model, module).load_state_dict(checkpoint[module])