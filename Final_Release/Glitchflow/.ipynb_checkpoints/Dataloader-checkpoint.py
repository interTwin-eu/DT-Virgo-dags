import torch
from torch.utils.data import random_split
from .config import device

import numpy as np
import mlflow

import itwinai
from itwinai.components import DataGetter,DataSplitter,DataProcessor,  monitor_exec

from typing import List, Optional
import os
import time
import datetime

from . Data import (TFrame,
                    save_tensor,
                    augment_data,augment_dataset,augment_list,
                   filter_rows_below_threshold,
                    filter_rows_above_threshold,
                   find_max,
                   normalize_ch_mean,YamlWrapper,gen_mask)



class ReadInfData(DataGetter):
    def __init__(self, data_root:str='./temp/',
                 f_means:str='means.pt',
                 f_backgr:str='background.pt',
                 f_test:str='test.pt',
                 uri:str="http://localhost:5005",
                 model_name:str='Unet_attention_trained',
                 model_version:str='latest'
                 ) -> None:
        
        
        
        """
        Class for reading data for inference
        
        data_root: path to data for inference. temp is the default directory created by setup.sh
        f_means(list): channel means tensor name
        f_backr(torch.tensor): background dataset tensor name
        f_test(TFrame): test dataset tensor name
        uri: mlflow uri
        model_name: model name
        model_version: model version 

        returns a list

        [means,background,test,model,gps,self.model_name]

        gps are read from the dataset(TFrame)
        
        """
        self.data_root=data_root
        self.f_means=f_means
        self.f_backgr=f_backgr
        self.f_test=f_test
        self.uri=uri
        self.model_name=model_name
        self.model_version=model_version
        
    def LoadTensor(self, f_name: str):
        """If the dataset is not given, load it from disk."""
        
        
        
        return torch.load(self.data_root+f_name,weights_only=False)
    
    
    
    def LoadTest(self, f_name: str):
        """If the dataset is not given, load it from disk."""
        
        
        tf=TFrame(0,[],[],[],{})
            
        tf.load(self.data_root+f_name,safeload=False)
        
            
        return tf
        
        
        

    @monitor_exec
    def execute(self) -> List:
        
        means=self.LoadTensor(self.f_means)#list
        background=self.LoadTensor(self.f_backgr)
        
        qtf=self.LoadTest(self.f_test)
        test=qtf.tdata
        
        gps=qtf.gps
        
        mlflow.set_registry_uri(self.uri)
        model= mlflow.pytorch.load_model(f"models:/{self.model_name}/{self.model_version}")
        
        
        
        
        
        
        return [means,background,test,model,gps,self.model_name]
    
    
    
    


    
    



    
    
#-----------------------------------------Classes for training-----------------------------------------------------------------------------------#    
class QTDatasetSplitter(DataSplitter):
    def __init__(
        self,
        train_proportion: int | float,
        
        validation_proportion: int | float = 0.0,
        
        
        rnd_seed: Optional[int] = 42,
        
        name: Optional[str] = None,
        images_dataset: Optional[str] = None,
        nev:int=100, #only for testing
        nchans:int=2 #only for testing
        
    ) -> List:
        """Class for splitting of smaller datasets. Use this class in the pipeline if the
        entire dataset can fit into memory.

        Args:
            train_proportion (int | float): _description_
            validation_proportion (int | float, optional): _description_. Defaults to 0.0.
            test_proportion (int | float, optional): _description_. Defaults to 0.0.
            rnd_seed (Optional[int], optional): _description_. Defaults to None.
            images_dataset (str, optional): path to dataset
            nev: numbers of rows to select.If 0 will be ignored
            nchans: number of channels to select   
            name (Optional[str], optional): _description_. Defaults to None.

         returns a list 
          [train_data, test_data,num_aux_channels,gps_list_test]

          num_aux_channels is the number of channels
          gps are the ids of the events selected for the test proportion of the dataset
          
        """
        test=1 - train_proportion
        super().__init__(train_proportion,validation_proportion,test , name)
        self.save_parameters(**self.locals2params(locals()))
        #self.validation_proportion =validation_propo 
        self.rnd_seed = rnd_seed
        self.images_dataset = images_dataset
        
        self.nev=nev
        self.nchans=nchans
       
        

    def get_or_load(self, dataset: Optional[torch.Tensor] = None):
        """If the dataset is not given, load it from disk."""
        if dataset is None:
            #LOG
            print("WARNING: QT dataset from disk.")
            
            tf=TFrame(0,[],[],[],{})
            
            tf.load(self.images_dataset)
            
            return tf
        return dataset

    @monitor_exec
    def execute(self, dataset: Optional[torch.Tensor] = None) -> List:
        
        qtf = self.get_or_load(dataset)
        
        dataset=qtf.tdata
        print(dataset.shape)
        gps_list=qtf.gps
        
        print(len(gps_list))
        print('Read dataset: ',dataset.shape)
        
        
        torch.manual_seed( self.rnd_seed)  # Choose any integer as the seed
        #data=dataset[:self.nev,:self.nchans,:,:]#remove after test
        
        if self.nev:
            data=dataset[:self.nev,:self.nchans,:,:]#remove after test 
            
        else:    
            data=dataset
        #gps_list=gps_list[:self.nev]
        #data=dataset
        print('Selected dataset')
        print(data.shape)
        
        
        num_aux_channels=data.shape[1]-1
        

        # Set split sizes: 90% for training, 10% for testing
        train_size = int(self.train_proportion * len(data))
        test_size = len(data) - train_size

        # Perform the train-test split with the fixed seed
        train_data_list, test_data_list = random_split(data, [train_size, test_size])


       # Convert the Subset objects back to tensors
        train_data = torch.stack([data[idx] for idx in train_data_list.indices])
        test_data = torch.stack([data[idx] for idx in test_data_list.indices])
        gps_list_test=[gps_list[idx] for idx in test_data_list.indices]
        
        
        print('test list')
        print(len(gps_list_test))


       # Check the final concatenated shapes
        print(f'{train_data.shape=}\n{test_data.shape=}')

        
        
        return [train_data, test_data,num_aux_channels,gps_list_test]
    
    
class QTProcessor(DataProcessor):
    def __init__(self, name: str | None = None,maxclamp: int | float = 10000,minclamp: int | float = 0,maxstrain:int=16,maxstrain_blw:int=10) -> List:
       
        """
        Args:
            name (str | None, optional): Defaults to None.
            
            maxclamp,minclamp: value for dataset clamp
            maxstrain: strain value for dataset filter from above
            maxstrain_blw: strain value for dataset filter from below

            returns a list

            [train_data_2d_norm,background_norm,test_data_2d_norm, norm_factor,num_aux_channels,channel_means,gps_selected]  

            
        """
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))#itwinai backend
        
        self.max_value=maxclamp
        self.min_value=minclamp
        self.max_strain=maxstrain
        self.max_strain_blw=maxstrain_blw

    @monitor_exec
    def execute(self, dataset:List) -> List:
        
        """Pre-process datasets: rearrange and normalize before training.

        
        """
        
       #AUGMENTATION
    
        num_aux_channels=dataset[2]
        gps_list=dataset[3]
        
        train_data_2d,test_data_2d=augment_dataset(dataset[0],dataset[1])
        
        print('Augmented dataset:')
        print(f'{train_data_2d.shape=}\n{test_data_2d.shape=}')
        
        del dataset
        
        
        #CLAMPING
        
        print('Clamp dataset:')
        
        
        train_data_2d_clamp=torch.clamp(train_data_2d, min=self.min_value,max=self.max_value)
        test_data_2d_clamp=torch.clamp(test_data_2d, min=self.min_value,max=self.max_value)
        try:
            background_tensor_clamp=torch.clamp(background_tensor, min=0,max=max_value)
        except:
            print('No background tensor')
            
            
        #gps_aug=augment_list(gps_list)
            
        
        #FILTERING
        
        print('Filtering dataset below treshold:')
        
        
        filtered_data_train_2d_below,mask_train=filter_rows_below_threshold(train_data_2d_clamp,
                                                                            gen_mask(self.max_strain_blw,
                                                                                 train_data_2d_clamp.shape[1],
                                                                                 self.max_value).to(device))
            
        filtered_data_test_2d_below, mask_test=filter_rows_below_threshold(test_data_2d_clamp,
                                                                           gen_mask(self.max_strain_blw,
                                                                                    train_data_2d_clamp.shape[1],
                                                                                    self.max_value).to(device))
            
         
        
        
        print('-Background')
        
        background=torch.cat((filtered_data_train_2d_below,filtered_data_test_2d_below))
        
        print(background.shape)
        
        del filtered_data_train_2d_below
        del filtered_data_test_2d_below
        
       
        
        
        
        print('Filtering dataset above treshold:')
        
        filtered_data_train_2d,_=filter_rows_above_threshold(train_data_2d_clamp,
                                                                            gen_mask(self.max_strain,
                                                                                 train_data_2d_clamp.shape[1],
                                                                                 self.min_value).to(device))
            
        filtered_data_test_2d, mask_test=filter_rows_above_threshold(test_data_2d_clamp,
                                                                           gen_mask(self.max_strain,
                                                                                    train_data_2d_clamp.shape[1],
                                                                                    self.min_value).to(device))
        
        gps_selected=[] 
        
        for i in range(len(mask_test)):
            
            if mask_test[i]:
                
                idx=int(i/5)
                #print(idx, i)
                gps_selected.append(gps_list[idx])
                
                
                
        
        
        print('Selected test data',len(gps_selected))
        
        
        
        print('-Train')
        print(filtered_data_train_2d.shape)
        print('-Test')
        print(filtered_data_test_2d.shape)
        
        
        
        
        #STAST AND NORMALIZATION
        print('Normalize and print statistics')
        

        #Filtered data
        max_train = find_max(filtered_data_train_2d)
        max_test = find_max(filtered_data_test_2d)


        # Flatten the tensor along the channel dimension
        flattened_tensor = max_train.view(-1, num_aux_channels+1)
        flattened_tensor_test = max_test.view(-1, num_aux_channels+1)

        # Convert tensor to numpy array
        numpy_array = flattened_tensor.numpy()
        numpy_array_test= flattened_tensor_test.numpy()
        
        channel_means = np.mean(numpy_array, axis=0)
        channel_means_test = np.mean(numpy_array_test, axis=0)
        channel_std = np.std(numpy_array, axis=0)
        channel_std_test = np.std(numpy_array_test, axis=0)
        
        print('\n\nTRAIN')
        for i, mean in enumerate(channel_means):
            print(f'Average of Channel {i+1} train: {mean}')
            print(f'std of Channel {i+1} train: {channel_std[i]}')
            print(f'-----------------------------------------')
        print('\n\n TEST')   
        for i, mean in enumerate(channel_means_test):
            print(f'Average of Channel {i+1} test: {mean}')
            print(f'STD of Channel {i+1} train: {channel_std_test[i]}')
            print(f'-----------------------------------------')
            
        norm_factor=torch.tensor(channel_means[0]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        
        train_data_2d_norm=normalize_ch_mean(filtered_data_train_2d,channel_means) #,channel_std
        #,channel_means,channel_std # not channel_means_test, it should be the same as train data
        test_data_2d_norm=normalize_ch_mean(filtered_data_test_2d,channel_means)
        #,channel_means,channel_std # not channel_means_test, it should be the same as train data
        
        background_norm=normalize_ch_mean(background,channel_means) 
        
        print(test_data_2d_norm.shape)
        print(train_data_2d_norm.shape)
        print(background_norm.shape)
        
        
        

        return [train_data_2d_norm,background_norm,test_data_2d_norm, norm_factor,num_aux_channels,channel_means,gps_selected]   
    
    



    
    

    
    
    

    
