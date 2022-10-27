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

def ising_energy_rbm(rbm_weights, rbm_visible_bias, rbm_hidden_bias, rbm_vis, rbm_hid):
    """
    Computes the energies produced by a randomly initialized Ising Model
    Using Parameters of an RBM. More detail is in:
    https://www.overleaf.com/5977645298fpvbhhnphxpy
    Energy:- rbm_vis^T rbm_weights rbm_hid + rbm_vis_bias ^ T rbm_vis + rbm_hid_bias ^ T rbm_hid + offset
    the offset term arises as we make the conversion from ising to rbm variables: V:{-1,1} -> {0,1}
    """
    # Calculate offset term
    ising_weights = rbm_weights/4
    ising_visible_bias = 0.5*rbm_visible_bias + 0.25*torch.sum(rbm_weights, dim=1)
    ising_hidden_bias = 0.5*rbm_hidden_bias + 0.25*torch.sum(rbm_weights, dim=0)
    w_sum = torch.sum(torch.sum(ising_weights, dim=0))
    v_bias_sum = torch.sum(ising_visible_bias, dim=0)
    h_bias_sum = torch.sum(ising_hidden_bias, dim=0)
    offset = w_sum - v_bias_sum - h_bias_sum
    
    # Device
    # preprocess weights (batchSize * nVis * nHid)
    rbm_weights = rbm_weights + torch.zeros((rbm_vis.size(0),) + rbm_weights.size(), device=rbm_vis.device)
    rbm_visible_bias = rbm_visible_bias.to(rbm_vis.device)
    rbm_hidden_bias = rbm_hidden_bias.to(rbm_hid.device)
    
    # preprocess from (batchSize * nVis) to (batchSize * 1 * nVis)
    vis = rbm_vis.unsqueeze(2).permute(0, 2, 1)
    # preprocess from (batchSize * nHid) to (batchSize * nHid * 1)
    hid = rbm_hid.unsqueeze(2)
    
    # compute the energy of the RBM
    rbm_energy = (torch.matmul(vis, torch.matmul(rbm_weights, hid)).reshape(-1) 
                + torch.matmul(rbm_vis, rbm_visible_bias)
                + torch.matmul(rbm_hid, rbm_hidden_bias))
    
    # rbm_energy+offset gives total energy
    return rbm_energy+offset


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

def batch_dwave_samples(response, qubit_idxs):
    """
    sampler.sample_ising() method returns a nested SampleSet structure
    with unique samples, energies and number of occurences stored in dict 
    
    Extract those values and construct a batch_size * (num_vis+num_hid) numpy array
    
    Returns:
        batch_samples : batch_size * (num_vis+num_hid) numpy array of samples collected by the DWave sampler
        batch_energies : batch_size * 1 numpy array of energies of samples
        
    UPDATE: There was a bug in which the dictionary was being processed. Thus bug has been fixed in this update
    """
    samples = []
    energies = []
    origSamples = []
    
    for sample_info in response.data():
        origSamples.extend([sample_info[0]]*sample_info[2]) # this is the original sample
        # the first step is to reorder
        origDict = sample_info[0] # it is a dictionary {0:-1,1:1,2:-1,3:-1,4:-1 ...} 
                                  # we need to rearrange it to {0:-1,1:1,2:-1,3:-1,132:-1 ...}
        keyorder = qubit_idxs
        reorderedDict = {k: origDict[k] for k in keyorder if k in origDict} # reorder dict
        
        uniq_sample = list(reorderedDict.values()) # one sample
        sample_energy = sample_info[1]
        num_occurences = sample_info[2]
        
        samples.extend([uniq_sample]*num_occurences)
        energies.extend([sample_energy]*num_occurences)
        
    batch_samples = np.array(samples)
    batch_energies = np.array(energies).reshape(-1)
        
    return batch_samples, batch_energies, origSamples

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
