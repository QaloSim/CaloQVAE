"""
Wandb compatible plotting functions
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from coffea import hist
from io import BytesIO
from PIL import Image
import wandb

from utils.plotting.HighLevelFeatures import HighLevelFeatures as HLF
# HLF_1_photons = HLF('photon', filename='/raid/javier/Datasets/CaloVAE/data/atlas/binning_dataset_1_photons.xml')
# HLF_1_pions = HLF('pion', filename='/raid/javier/Datasets/CaloVAE/data/atlas/binning_dataset_1_pions.xml')
# HLF_1_photons.relevantLayers = [1,2,3,4,5]
# HLF_1_pions.relevantLayers = [1,2,3,4,5,6,7]
# HLF_1_electron = HLF('electron', filename='/raid/javier/Datasets/CaloVAE/data/atlas_dataset2and3/binning_dataset_2.xml')
# HLF_1_electron.relevantLayers = [5,10,15,20,25,30,35,40,44]
# HLF_1_photons = HLF('photon', filename='/fast_scratch/QVAE/data/atlas/binning_dataset_1_photons.xml')
# HLF_1_pions = HLF('pion', filename='/fast_scratch/QVAE/data/atlas/binning_dataset_1_pions.xml')

_NORM_LIST = [LogNorm(vmax=10000, vmin=0.1), LogNorm(vmax=10000 , vmin=0.1), LogNorm(vmax=10, vmin=0.1)]

# def plot_calo_images(layer_images, particle='pion'):
#     image_list = []
#     for idx in range(layer_images[0].shape[0]):
#         image_list.append(plot_atlas_image2(layer_images[0][idx,:], particle))      
# #         image_list.append(plot_atlas_image([layer_image[idx] for layer_image in layer_images]))
#     return image_list

def plot_calo_images(layer_images, HLF):
    image_list = []
    for idx in range(layer_images[0].shape[0]):
        image_list.append(plot_atlas_image2(layer_images[0][idx,:], HLF))
    return image_list
        
def plot_calo_image(image):
    fig, ax = plt.subplots(nrows=1, ncols=len(image), figsize=(12, 4))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.45, hspace=None)
    
    for layer, axe in enumerate(fig.axes):
        im = axe.imshow(image[layer], aspect='auto', origin='upper', norm=_NORM_LIST[layer])
        axe.set_title('Layer ' + str(layer), fontsize=10)
        axe.tick_params(labelsize=10)
        
        axe.set_yticks(np.arange(0, image[layer].shape[0], 1))
        axe.set_xlabel(r'$\phi$ Cell ID', fontsize=10)
        axe.set_ylabel(r'$\eta$ Cell ID', fontsize=10)
        
        if layer == 0:
            axe.set_xticks(np.arange(0, image[layer].shape[1], 10))
        else:
            axe.set_xticks(np.arange(0, image[layer].shape[1], 1))
            
        cbar = fig.colorbar(im, ax=axe)
        cbar.set_label('Energy, (MeV)', fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=500)
    buf.seek(0)
    image = wandb.Image(Image.open(buf))
    buf.close()
    plt.close(fig)
        
    return image

def plot_atlas_image(image):
    fig, ax = plt.subplots(nrows=1, ncols=len(image), figsize=(12, 4))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.45, hspace=None)
    
    for layer, axe in enumerate(fig.axes):
        im = axe.imshow(image[layer].reshape((8,46)), aspect='auto', origin='upper', norm=_NORM_LIST[layer])
        axe.set_title('Layer ' + str(layer), fontsize=10)
        axe.tick_params(labelsize=10)
        
#         axe.set_yticks(np.arange(0, image[layer].shape[0], 1))
        axe.set_xlabel(r'$\phi$ Cell ID', fontsize=10)
        axe.set_ylabel(r'$\eta$ Cell ID', fontsize=10)
        
#         if layer == 0:
#             axe.set_xticks(np.arange(0, image[layer].shape[1], 10))
#         else:
#             axe.set_xticks(np.arange(0, image[layer].shape[1], 1))
            
        cbar = fig.colorbar(im, ax=axe)
        cbar.set_label('Energy, (MeV)', fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=500)
    buf.seek(0)
    image = wandb.Image(Image.open(buf))
    buf.close()
    plt.close(fig)
        
    return image

def plot_atlas_image2(image0, HLF):
    # if particle == 'pion':
    #     image = HLF_1_pions.DrawSingleShower(image0, filename=None)
    # elif particle == 'photon':
    #     image = HLF_1_photons.DrawSingleShower(image0, filename=None)
    # elif particle == 'electron':
    #     image = HLF_1_electron.DrawSingleShower(image0, filename=None)
    image = HLF.DrawSingleShower(image0, filename=None)
    return image

# def plot_atlas_image2(image0, particle='pion'):
#     if particle == 'pion':
#         image = HLF_1_pions.DrawSingleShower(image0, filename=None)
#     elif particle == 'photon':
#         image = HLF_1_photons.DrawSingleShower(image0, filename=None)
#     elif particle == 'electron':
#         image = HLF_1_electron.DrawSingleShower(image0, filename=None)
# #     fig, ax = plt.subplots(nrows=1, ncols=len(image), figsize=(12, 4))
# #     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.45, hspace=None)
    
# #     for layer, axe in enumerate(fig.axes):
# #         im = axe.imshow(image[layer].reshape((8,46)), aspect='auto', origin='upper', norm=_NORM_LIST[layer])
# #         axe.set_title('Layer ' + str(layer), fontsize=10)
# #         axe.tick_params(labelsize=10)
        
# # #         axe.set_yticks(np.arange(0, image[layer].shape[0], 1))
# #         axe.set_xlabel(r'$\phi$ Cell ID', fontsize=10)
# #         axe.set_ylabel(r'$\eta$ Cell ID', fontsize=10)
        
# # #         if layer == 0:
# # #             axe.set_xticks(np.arange(0, image[layer].shape[1], 10))
# # #         else:
# # #             axe.set_xticks(np.arange(0, image[layer].shape[1], 1))
            
# #         cbar = fig.colorbar(im, ax=axe)
# #         cbar.set_label('Energy, (MeV)', fontsize=10)
# #         cbar.ax.tick_params(labelsize=10)
        
# #     buf = BytesIO()
# #     plt.savefig(buf, format='png', dpi=500)
# #     buf.seek(0)
# #     image = wandb.Image(Image.open(buf))
# #     buf.close()
# #     plt.close(fig)
        
#     return image
