"""Load up the MNIST data."""
from copy import deepcopy
import numpy as np
import torch
import h5py
from torch.utils.data import random_split, Dataset, Subset
from torchvision import transforms

from CaloQVAE import logging
logger = logging.getLogger(__name__)

#class wrapper containing Calorimeter images and returning energy as target
class CaloImage(object):
    def __init__(self, image, layer):        
        self._image=image
        self._layer=layer
        self._input_size=self._image[0].view(-1).size()[0]
        self._input_dimension=self._image[0].shape

    def __len__(self):
        return len(self._image)
    
    #TODO this normalises the input data to [0,1]
    #needs to be changed for proper energy deposits.
    def normalise(self, img):
        minVal=img.view(-1,self._input_size).min(1,keepdim=True)[0]
        maxVal=img.view(-1,self._input_size).max(1,keepdim=True)[0]
        #if img all 0
        if abs(maxVal)>0:
            img-=minVal
            img/=maxVal
        return img

    def _get_image(self,idx):
        return self._image[idx]
        #return self.normalise(self._image[idx])
    
    def get_flattened_input_size(self):
        return self._input_size
    
    def get_input_dimension(self):
        return self._input_dimension

#contains images for all layers and the energy
class CaloImageContainer(Dataset):
    
    def __init__(self, particle_type=None, input_data=None, layer_subset=[]):
        self._particle_type=particle_type

        try:
            self._dataset_size=len(input_data["voxels"])
        except:
            try:
                self._dataset_size=len(input_data["layer_0"])
            except: self._dataset_size=len(input_data["showers"])
        #dictionary of all calo images - keys are layer names
        self._images=None
        #true energy of the jets per event (same for all layers)
        self._true_energies=None
        #energy deposited outside the calorimeter layers - 1 value per layer.
#         self._overflow_energies=None
        #the available events are split to train, test or val
        self._event_label=None

        #only use selected layers
        self._layer_subset=layer_subset

        #this will be used to steer train/test/val splitting
        self._indices = None
    
    def create_subset(self, idx_list, label):
        assert isinstance(idx_list,list), "Indices must be list"
        subset=deepcopy(self)
        subset._indices=idx_list
        subset._event_label=label
        return subset

    def __len__(self):
        return len(self._indices) if self._indices else self._dataset_size

    #pytorch dataloader needs this method
    def __getitem__(self, ordered_idx):
        #indexing the shuffled list of event indices of our full dataset
        #creates random event selection
        rnd_idx=self._indices[ordered_idx]

        norm_true_energy=self._true_energies[rnd_idx]
#         norm_overflow_energy=self._overflow_energies[rnd_idx]
        
        images=[]
        #if we request a subset of the calorimeter layers only
        if len(self._layer_subset)>0:
            for l in self._layer_subset:
                if len(l)>0:
                    images.append(self._images[l]._get_image(rnd_idx))
        else:
            images=[img._get_image(rnd_idx) for l, img in self._images.items()]
#         return images, (norm_true_energy, norm_overflow_energy)
        return images, (norm_true_energy, 0)
    
    def get_flattened_input_sizes(self):
        sizes=[]
        
        layers=self._layer_subset if len(self._layer_subset)>0 else self._images.keys()
        for l in layers:
            #TODO remove this if once working with HYDRA
            if len(l)>0:
                sizes.append(self._images[l].get_flattened_input_size())
        return sizes
    
    def get_input_dimensions(self):
        dim=[]
        for l,img in self._images.items():
            dim.append(img.get_input_dimension())
        return dim
    
    def get_dataset_size(self):
        """Returns number of events in layer_0.
        """
        return self._dataset_size

    def process_data(self, input_data):
        calo_images={}
        for key, item in input_data.items():
            #do not process energies here
            if key.lower() in ["energy","overflow"]: continue
            ds=input_data[key][:]
            #convert df to CaloImage
            calo_images[key]=CaloImage(image=torch.Tensor(ds),layer=key)

        self._images=calo_images
        try:
            self._true_energies=input_data["energy"][:]
        except:
            self._true_energies=input_data["incident_energies"][:]
        try:
            self._overflow_energies=input_data["overflow"][:]
        except:
            pass

def get_atlas_datasets(inFiles={}, particle_type=["pions1"], layer_subset=[],
                      frac_train_dataset=0.6, frac_test_dataset=0.2):

    #read in all input files for all jet types and layers
    dataStore={}
    for key,fpath in inFiles.items():     
        in_data=h5py.File(fpath,'r')
        #for each particle_type, create a Container instance for our needs   
        dataStore[key]=CaloImageContainer(  particle_type=key,
                                            input_data=in_data,
                                            layer_subset=layer_subset)
        #convert image dataframes to tensors and get energies
        dataStore[key].process_data(input_data=in_data)

    assert len(particle_type)==1, f"Currently one particle type at a time\
         can be retrieved. Requested {particle_type}"
    ptype=particle_type[0]

    #let's split our datasets
    #get total num evts
    num_evts_total=dataStore[ptype].get_dataset_size()
    
    #create a sequential list of indices
    idx_list = [i for i in range(0, num_evts_total)]
    
    # compute number of split evts from fraction
    num_evts_train = int(frac_train_dataset*num_evts_total)
    num_evts_test = int(frac_test_dataset*num_evts_total)

    #create lists of split indices
    train_idx_list = idx_list[:num_evts_train]
    test_idx_list = idx_list[num_evts_train:(num_evts_train+num_evts_test)]
    val_idx_list = idx_list[(num_evts_train+num_evts_test):]

    train_dataset = dataStore[ptype].create_subset(idx_list=train_idx_list, label="train")
    test_dataset = dataStore[ptype].create_subset(idx_list=test_idx_list, label="test")
    val_dataset = dataStore[ptype].create_subset(idx_list=val_idx_list, label="val")

    return train_dataset, test_dataset, val_dataset
