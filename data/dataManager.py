"""
Data Manager

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
import torch
import numpy as np
import joblib
# from torch.utils.data import DataLoader

from CaloQVAE import logging
logger = logging.getLogger(__name__)

from data.mnist import get_mnist_datasets
from data.calo import get_calo_datasets
from data.atlas import get_atlas_datasets

import h5py
from torch.utils.data import TensorDataset, DataLoader

# Constants
_EPSILON = 1e-2

class DataManager(object):
    def __init__(self,train_loader=None,test_loader=None,val_loader=None, cfg=None):
        self._config=cfg
        
        self._train_loader=train_loader
        self._test_loader=test_loader
        self._val_loader=val_loader

        #this is a list of tensor.shape tuples (i.e.[(28,28)] for MNIST) 
        self._input_dimensions=None
        #list of flattened tensor.shape tuples (i.e. [784] for mnist)
        self._flat_input_sizes=None 

        self._train_dataset_means=None
        
        # Variables to be used in the scaling and inverse scaling
        self._amin_array = None
        self._transformer = None
        return

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def test_loader(self):
        return self._test_loader
    
    @property
    def val_loader(self):
        return self._val_loader

    def init_dataLoaders(self):
        logger.info("Loading Data")

        if not self._config.load_data_from_pkl:
            train_loader,test_loader,val_loader=self.create_dataLoader()
        else:
            #TODO
            train_loader,test_loader,val_loader,_,__=self.load_from_file()
        
        assert bool(train_loader and test_loader and val_loader), "Failed to set up data_loaders"
        
        self._train_loader=train_loader
        self._test_loader=test_loader
        self._val_loader=val_loader
        return

    def get_train_dataset_mean(self):
        return self._train_dataset_mean

    def get_input_dimensions(self):
        return self._input_dimensions
    
    def get_flat_input_size(self):
        return self._flat_input_sizes

    def _set_input_dimensions(self):
        assert self._train_loader is not None, "Trying to retrieve datapoint from empty train loader"
        self._input_dimensions=self._train_loader.dataset.get_input_dimensions()
    
    def _set_flattened_input_sizes(self):
        assert self._train_loader is not None, "Trying to retrieve datapoint from empty train loader"
        self._flat_input_sizes=self._train_loader.dataset.get_flattened_input_sizes()

    def _set_train_dataset_mean(self):
        #TODO should this be the mean over the current batch only?
        #returns mean of dataset as list
        assert self._train_loader is not None, "Trying to retrieve datapoint from empty train loader"
        
        in_sizes=self.get_flat_input_size()
        imgPerLayer={}	
        
        #create an entry for each layer
        for i in range(0,len(in_sizes)):
            imgPerLayer[i]=[]	

        for i, (data, _) in enumerate(self._train_loader.dataset):
            #loop over all layers
            for l,d in enumerate(data):	
                imgPerLayer[l].append(d.view(-1,in_sizes[l]))
        means=[]
        for l, imgList in imgPerLayer.items():
            means.append(torch.mean(torch.stack(imgList),dim=0))

        self._train_dataset_mean=means
        
    def _set_amin_array(self):
        assert self._config.data.scaler_amin
        with open(self._config.data.scaler_amin, 'rb') as f:
            self._amin_array = np.load(f)
        
    def _set_transformer(self):
        assert self._config.data.scaler_path
        self._transformer = joblib.load(self._config.data.scaler_path)

    def pre_processing(self):
        if not self._config.load_data_from_pkl:
            self._set_input_dimensions()
            self._set_flattened_input_sizes()
            self._set_train_dataset_mean()
            if self._config.data.scaled:
                self._set_transformer()
                self._set_amin_array()
        else:
            #TODO load from file
            raise NotImplementedError
            # _,__,input_dimensions,train_ds_mean=self.load_from_file()
            # self._input_dimensions=input_dimensions
            # self._train_dataset_mean=train_ds_mean

    def create_dataLoader(self):
        assert abs(self._config.data.frac_train_dataset-1)>=0, "Cfg option frac_train_dataset must be within (0,1]"
        assert abs(self._config.data.frac_test_dataset-0.99)>1.e-5, "Cfg option frac_test_dataset must be within (0,99]. 0.01 minimum for validation set"

        if self._config.data.data_type.lower()=="mnist":
            train_dataset,test_dataset,val_dataset=get_mnist_datasets(
                frac_train_dataset=self._config.data.frac_train_dataset,
                frac_test_dataset=self._config.data.frac_test_dataset, 
                binarise=self._config.data.binarise_dataset,
                input_path=self._config.data.mnist_input)

        elif self._config.data.data_type.lower()=="calo":
            inFiles={
                'gamma':    self._config.data.calo_input_gamma,
                'eplus':    self._config.data.calo_input_eplus,        
                'piplus':   self._config.data.calo_input_piplus         
            }

            train_dataset,test_dataset,val_dataset=get_calo_datasets(
                inFiles=inFiles,
                particle_type=[self._config.data.particle_type],
                layer_subset=self._config.data._layers,
                frac_train_dataset=self._config.data.frac_train_dataset,
                frac_test_dataset=self._config.data.frac_test_dataset, 
                )
            
        elif self._config.data.data_type.lower()=="atlas":
            inFiles={
            'photon1':    self._config.data.atlas_input_photon1,
                'photonEn0':    self._config.data.atlas_input_photonEn0,
                'photonEn1':    self._config.data.atlas_input_photonEn1,
                'photonEn2':    self._config.data.atlas_input_photonEn2,
                'photonEn3':    self._config.data.atlas_input_photonEn3,
                'photonEn4':    self._config.data.atlas_input_photonEn4,
                'photonEn5':    self._config.data.atlas_input_photonEn5,
                'photonEn6':    self._config.data.atlas_input_photonEn6,
                'photonEn7':    self._config.data.atlas_input_photonEn7,
            'pion1':   self._config.data.atlas_input_pion1,
                'pionEn0':   self._config.data.atlas_input_pionEn0,
                'pionEn1':   self._config.data.atlas_input_pionEn1,
                'pionEn2':   self._config.data.atlas_input_pionEn2,
                'pionEn3':   self._config.data.atlas_input_pionEn3,
                'pionEn4':   self._config.data.atlas_input_pionEn4,
                'pionEn5':   self._config.data.atlas_input_pionEn5,
                'pionEn6':   self._config.data.atlas_input_pionEn6,
                'pionEn7':   self._config.data.atlas_input_pionEn7,
            'electron-ds2': self._config.data.atlas_input_electron,
        }

            train_dataset,test_dataset,val_dataset=get_atlas_datasets(
                inFiles=inFiles,
                particle_type=[self._config.data.particle_type],
                layer_subset=self._config.data._layers,
                frac_train_dataset=self._config.data.frac_train_dataset,
                frac_test_dataset=self._config.data.frac_test_dataset, 
                )
                
        #create the DataLoader for the training dataset
        train_loader=DataLoader(   
            train_dataset,
            batch_size=self._config.engine.n_train_batch_size, 
            num_workers=self._config.num_workers,
            shuffle=True)

        #create the DataLoader for the testing/validation datasets
        #set batch size to full test/val dataset size - limitation only by hardware
        test_loader = DataLoader(
            test_dataset,
            batch_size=self._config.engine.n_test_batch_size, 
            num_workers=self._config.num_workers,
            shuffle=False)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._config.engine.n_valid_batch_size, 
            num_workers=self._config.num_workers,
            shuffle=False)

        logger.info("{0}: {2} events, {1} batches".format(train_loader,len(train_loader),len(train_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(test_loader,len(test_loader),len(test_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(val_loader,len(val_loader),len(val_loader.dataset)))

        return train_loader,test_loader,val_loader
    
    def load_from_file(self):
        #To speed up chain. Preprocessing involves loop over data for normalisation.
        #Load that data already prepped.
        import pickle
        with open(self._config.pre_processed_input_file, "rb") as dataFile:
            train_loader    =pickle.load(dataFile)
            test_loader     =pickle.load(dataFile)
            input_dimensions =pickle.load(dataFile)
            train_ds_mean   =pickle.load(dataFile)
        return train_loader, test_loader, input_dimensions, train_ds_mean
    
    def inv_transform(self, data):
        """
        Applies inverse transformation to standard scaling
        
        Args:
            data - np array (num_examples * num_features)
            
        Returns:
            nparr - Inverse transformed np array (num_examples * num_features)
        """
        # nparr = np.where(data > 0., data, np.inf)
        nparr = np.where(data > 0., data, np.nan)
#         logger.info(nparr.shape)
        # print(nparr.shape)
        for j in range(nparr.shape[1]):
            amin = self._amin_array[j]
            if amin < 0. and not np.isnan(amin) and not np.isinf(amin):
                nparr[:, j] -= _EPSILON
                nparr[:, j] += amin
                
        nparr = self._transformer.inverse_transform(nparr)
        # nparr = np.where(np.isinf(nparr), 0., nparr)
        nparr = np.where(np.isnan(nparr), 0., nparr)
        
        return nparr
    

class DataManagerBeta(object):
    def __init__(self,train_loader=None,test_loader=None,val_loader=None, cfg=None):
        self._config=cfg
        
        self._train_loader=train_loader
        self._test_loader=test_loader
        self._val_loader=val_loader

        #this is a list of tensor.shape tuples (i.e.[(28,28)] for MNIST) 
        self._input_dimensions=None
        #list of flattened tensor.shape tuples (i.e. [784] for mnist)
        self._flat_input_sizes=None 

        self._train_dataset_means=None
        
        # Variables to be used in the scaling and inverse scaling
        self._amin_array = None
        self._transformer = None
        self._set_amin_array()
        self._set_transformer()
        # return
        self.particle_type = [self._config.data.particle_type]
        self.frac_train_dataset=self._config.data.frac_train_dataset
        self.frac_test_dataset=self._config.data.frac_test_dataset
        self.load_dataset_directories()
        _,_,_ = self.create_dataLoader()
        
        
    def load_dataset_directories(self):
        if self._config.data.data_type.lower()=="atlas":
            self.inFiles={
            'photon1':    self._config.data.atlas_input_photon1,
                'photonEn0':    self._config.data.atlas_input_photonEn0,
                'photonEn1':    self._config.data.atlas_input_photonEn1,
                'photonEn2':    self._config.data.atlas_input_photonEn2,
                'photonEn3':    self._config.data.atlas_input_photonEn3,
                'photonEn4':    self._config.data.atlas_input_photonEn4,
                'photonEn5':    self._config.data.atlas_input_photonEn5,
                'photonEn6':    self._config.data.atlas_input_photonEn6,
                'photonEn7':    self._config.data.atlas_input_photonEn7,
            'pion1':   self._config.data.atlas_input_pion1,
                'pionEn0':   self._config.data.atlas_input_pionEn0,
                'pionEn1':   self._config.data.atlas_input_pionEn1,
                'pionEn2':   self._config.data.atlas_input_pionEn2,
                'pionEn3':   self._config.data.atlas_input_pionEn3,
                'pionEn4':   self._config.data.atlas_input_pionEn4,
                'pionEn5':   self._config.data.atlas_input_pionEn5,
                'pionEn6':   self._config.data.atlas_input_pionEn6,
                'pionEn7':   self._config.data.atlas_input_pionEn7,
            'electron-ds2': self._config.data.atlas_input_electron,
        }
        
    def load_data(self):
        
        #read in all input files for all jet types and layers
        datastore={}
        for key,fpath in self.inFiles.items(): 
            if key in self.particle_type: 
                with h5py.File(fpath, 'r') as file:
                    # List all groups
                    # print("Keys: %s" % file.keys())
                    for other_key in file.keys():
                        datastore[other_key] = torch.tensor(file[other_key][:]).float()
                
                # in_data=h5py.File(fpath,'r')
                # #for each particle_type, create a Container instance for our needs   
                # dataStore[key]=CaloImageContainer(  particle_type=key,
                #                                     input_data=in_data,
                #                                     layer_subset=layer_subset)
                # #convert image dataframes to tensors and get energies
                # dataStore[key].process_data(input_data=in_data)

        assert len(self.particle_type)==1, f"Currently one particle type at a time\
             can be retrieved. Requested {self.particle_type}"
        ptype=self.particle_type[0]
        self._flat_input_sizes = datastore['showers'].shape[1]
        print(self._flat_input_sizes)

        #let's split our datasets
        #get total num evts
        num_evts_total=datastore['showers'].shape[0]
        

        #create a sequential list of indices
        # idx_list = [i for i in range(0, num_evts_total)]

        # compute number of split evts from fraction
        num_evts_train = int(self.frac_train_dataset*num_evts_total)
        num_evts_test = int(self.frac_test_dataset*num_evts_total)

        #create lists of split indices
        # train_idx_list = idx_list[:num_evts_train]
        # test_idx_list = idx_list[num_evts_train:(num_evts_train+num_evts_test)]
        # val_idx_list = idx_list[(num_evts_train+num_evts_test):]

        # train_dataset = dataStore[ptype].create_subset(idx_list=train_idx_list, label="train")
        # test_dataset = dataStore[ptype].create_subset(idx_list=test_idx_list, label="test")
        # val_dataset = dataStore[ptype].create_subset(idx_list=val_idx_list, label="val")
        
        self.train_dataset =  TensorDataset(datastore['showers'][:num_evts_train,:], datastore['incident_energies'][:num_evts_train,:])
        self.test_dataset =  TensorDataset(datastore['showers'][num_evts_train:num_evts_train+num_evts_test,:], datastore['incident_energies'][num_evts_train:num_evts_train+num_evts_test,:])
        self.val_dataset =  TensorDataset(datastore['showers'][num_evts_train+num_evts_test:,:], datastore['incident_energies'][num_evts_train+num_evts_test:,:])

        return self.train_dataset, self.test_dataset, self.val_dataset
        
    def create_dataLoader(self):
        assert abs(self._config.data.frac_train_dataset-1)>=0, "Cfg option frac_train_dataset must be within (0,1]"
        assert abs(self._config.data.frac_test_dataset-0.99)>1.e-5, "Cfg option frac_test_dataset must be within (0,99]. 0.01 minimum for validation set"

        self.load_data()
                
        #create the DataLoader for the training dataset
        train_loader=DataLoader(   
            self.train_dataset,
            batch_size=self._config.engine.n_train_batch_size, 
            num_workers=self._config.num_workers,
            shuffle=True)

        #create the DataLoader for the testing/validation datasets
        #set batch size to full test/val dataset size - limitation only by hardware
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self._config.engine.n_test_batch_size, 
            num_workers=self._config.num_workers,
            shuffle=False)
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self._config.engine.n_valid_batch_size, 
            num_workers=self._config.num_workers,
            shuffle=False)

        logger.info("{0}: {2} events, {1} batches".format(train_loader,len(train_loader),len(train_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(test_loader,len(test_loader),len(test_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(val_loader,len(val_loader),len(val_loader.dataset)))
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        return train_loader,test_loader,val_loader
    
    def get_flat_input_size(self):
        return self._flat_input_sizes
    
    def _set_amin_array(self):
        assert self._config.data.scaler_amin
        with open(self._config.data.scaler_amin, 'rb') as f:
            self._amin_array = np.load(f)
            
    def _set_transformer(self):
        assert self._config.data.scaler_path
        self._transformer = joblib.load(self._config.data.scaler_path)
    
    def inv_transform(self, data):
        """
        Applies inverse transformation to standard scaling
        
        Args:
            data - np array (num_examples * num_features)
            
        Returns:
            nparr - Inverse transformed np array (num_examples * num_features)
        """
        # nparr = np.where(data > 0., data, np.inf)
        nparr = np.where(data > 0., data, np.nan)
#         logger.info(nparr.shape)
        
        for j in range(nparr.shape[1]):
            amin = self._amin_array[j]
            if amin < 0. and not np.isnan(amin) and not np.isinf(amin):
                nparr[:, j] -= _EPSILON
                nparr[:, j] += amin
                
        nparr = self._transformer.inverse_transform(nparr)
        # nparr = np.where(np.isinf(nparr), 0., nparr)
        nparr = np.where(np.isnan(nparr), 0., nparr)
        
        return nparr