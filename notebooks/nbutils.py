# Helper functions for notebooks
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
# Extra imports for image and data processing
from PIL import Image, ImageDraw
import json

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


def plot_energies(energies1, energies2, binwidth=1, beta=None, save_image=False):
    """
    Plot the energies of the samples produced by the histograms   
    UPDATE: bin now found using bin boundaries
    """
    fig, ax = plt.subplots(figsize=(40, 16))
    data = np.concatenate((energies1,energies2), axis=0)
    bins =  np.arange(min(data), max(data) + binwidth, binwidth)
    ax.hist(energies1, bins=bins, label = "dwave energies")  
    ax.hist(energies2, bins=bins, label = "auxiliary RBM energies", alpha=0.5) 
    plt.legend(loc='upper right', fontsize=30)
    if (beta!=None):
        plt.title(r'$\beta_{eff}^{*} = $'+format(beta,'.3f'), fontsize = 60)
    ax.set_xlabel("Energy", fontsize=60)
    ax.set_ylabel("Frequency", fontsize=60)
    
    ax.tick_params(axis='both', which='major', labelsize=60)
    ax.grid(True)
    
    if (save_image==False): 
        plt.show()
    plt.close()
    return fig


def save_run_info(run_info, ising_weights, ising_vbias, ising_hbias, aux_crbm_energy_exps, dwave_energies, betas, base_dir=None):
    """
    Inputs are the parameters whose infromation we want to save
    run_info is a string which is customised by the user to store
    custom information regarding a particular run.
    Data is saved in notebooks/Beta_estimation_data/beta_run_info
    """
    if base_dir==None:
        folder_dir = 'notebooks/Beta_estimation_data'
        if os.path.exists(folder_dir) == False:
            os.mkdir(folder_dir)
        base_dir = 'notebooks/Beta_estimation_data/beta_'+run_info
        if os.path.exists(base_dir) == False:
            os.mkdir(base_dir)
    weight_dir = base_dir+'/'+'ising_weights.pt'
    vbias_dir = base_dir+'/'+'ising_vbias.pt'
    hbias_dir = base_dir+'/'+'ising_hbias.pt'
    classical_energies_dir = base_dir+'/'+'aux_crbm_energy_exps.pt'
    dwave_energies_dir = base_dir+'/'+'dwave_energies.pt'
    betas_dir = base_dir+'/betas.pt'

    torch.save(ising_weights, weight_dir)
    torch.save(ising_vbias, vbias_dir)
    torch.save(ising_hbias, hbias_dir)
    torch.save(aux_crbm_energy_exps, classical_energies_dir)
    torch.save(dwave_energies, dwave_energies_dir)
    torch.save(betas, betas_dir)


def recover_saved_parameters(run_info, nb_data=1):
    """
    This should be used after saving information using save_run_info function
    Returns: ising_weights, ising_vbias, ising_hbias, aux_crbm_energy_exps, 
    dwave_energies, betas of a given run
    
    nb_data=1 (default) means the data files are saved in the notebook directory and
    nb_data = ELSE means data files are saved in run_info directory which must be hard coded.
    """
    if (nb_data==1):
        base_dir = 'notebooks/Beta_estimation_data/beta_'+run_info
    else:
        base_dir = run_info
    ising_weights = torch.load(base_dir+'/'+'ising_weights.pt')
    ising_vbias = torch.load(base_dir+'/'+'ising_vbias.pt')
    ising_hbias = torch.load(base_dir+'/'+'ising_hbias.pt')
    aux_crbm_energy_exps = torch.load(base_dir+'/'+'aux_crbm_energy_exps.pt')
    dwave_energies = torch.load(base_dir+'/'+'dwave_energies.pt')
    betas = torch.load(base_dir+'/'+'betas.pt')
    return ising_weights, ising_vbias, ising_hbias, aux_crbm_energy_exps, dwave_energies, betas


def save_energy_plots(run_info, dwave_energies, aux_crbm_energy_exps, betas, generate_gif=True, duration=1350, base_dir=None):
    """
    This saves energy plots using the plot_energies helper function.
    A GIF is also generated for convenience. 
    A smaller duration increases frequency of the GIF
    """
    image_dir=base_dir
    if base_dir==None:
        base_dir = 'notebooks/Beta_estimation_data/beta_'+run_info
        image_dir = base_dir+'/plots'
    # if os.path.exists(base_dir) == False:
    #    print("Incorrect run info")
    #    return 0
    if os.path.exists(image_dir) == False:
        os.mkdir(image_dir)
    for i in range(len(dwave_energies)):
        str_beta = format(betas[i], '.3f') # convert to string in 3 decimal places
        fig = plot_energies(dwave_energies[i], aux_crbm_energy_exps, 1, beta=betas[i], save_image=True) # make sure to add true false stuff
        image_file = image_dir+'/beta='+str_beta+'_fig_'+str(i+1)+'.jpg'
        fig.savefig(image_file)
        plt.close()
    print("Plots saved in {0}".format(image_dir))
    
    if (generate_gif==True):
        images = []
        for i in range(len(dwave_energies)):
            str_beta = format(betas[i], '.3f')
            images.append(Image.open(image_dir+'/beta='+str_beta+'_fig_'+str(i+1)+'.jpg'))

        images[0].save(image_dir+'/beta_energies.gif',
                       save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)
        print("GIF saved in {0}".format(image_dir))


def initialize_ising(n_vis, n_hid, nmean = None, std=None, wlim=None, hlim=None):
    """
    This function randomly initializes an Ising model with random J and h in the given ranges
    Inputs: nunmbers of visible and hidden nodes
    Output: Ising Weights and Biases
    * UPDATE: J is now drawn from a Gaussian distribution instead of a Uniform distribution
              Custom std,mean and limits can now added for J (i.e. J can be uniform/normal
              depending on input parameters)
    * TO-DO : Add option to draw h from a normal distribution too
    """
    if (wlim!=None and std==None and nmean==None):
        ising_weights = torch.nn.Parameter((wlim[1]-wlim[0])*torch.rand(n_vis, n_hid) + wlim[0], requires_grad=False) 
    elif (wlim==None and std!=None and nmean!=None):
        ising_weights = torch.normal(nmean, std, size=(n_vis, n_hid))
    else:
        print("Incorrect/Insufficent inputs to initialize Ising Model")
        return 0
    ising_vbias = torch.nn.Parameter((hlim[1]-hlim[0])*torch.rand(n_vis)+hlim[0], requires_grad=False)
    ising_hbias = torch.nn.Parameter((hlim[1]-hlim[0])*torch.rand(n_hid)+hlim[0], requires_grad=False)
    return ising_weights, ising_vbias, ising_hbias
