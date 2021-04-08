from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import math

from src.preprocessing import *
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


import os
import torch
import pandas as pd
import numpy as np
import math
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class EmersonDataset(Dataset):
    """
    Dataset class for data from the Emerson study (batch 1)
    """
    def __init__(self, path, max_len = 23, scale_by_freq = False):#, tokenizer):
        emerson = pd.read_csv(path, sep='\t', header=0, usecols = ['amino_acid','rel_freq','hla_a1','hla_a2','hla_b1','hla_b2','filename','len'])
        emerson = emerson.query('len<=@max_len')
        self.seq = emerson.amino_acid.values
        self.max_len = max_len
        self.rel_freq = emerson.rel_freq.values
        self.hla_a1 = emerson.hla_a1.values
        self.hla_a2 = emerson.hla_a2.values
        self.hla_b1 = emerson.hla_b1.values
        self.hla_b2 = emerson.hla_b2.values
        self.patients = emerson.filename.values
        self.unique = np.unique(self.patients)
        self.len = len(emerson)
        #self.tokenizer = tokenizer
        self.scale = scale_by_freq


    def __getitem__(self,index):
        #Gets the label for each HLA gene
        seqs = self.seq[index]
        label_a, label_b = get_all_label(self.hla_a1[index],
                                       self.hla_a2[index],
                                       self.hla_b1[index],
                                       self.hla_b2[index])
        
        #Different handling for a single element
        if type(seqs) == str: 
            label_a = label_a.unsqueeze(0)
            label_b = label_b.unsqueeze(0)
            encoded = pad_seq(onehot_aa(seqs), self.max_len).unsqueeze(0)
            if self.scale ==True:
                scale = self.rel_freq[index]
                encoded = encoded * scale
        #Multi element (batch)
        else: 
            encoded = onehot_batch(seqs, self.max_len)
            if self.scale ==True:
                scale = self.rel_freq[index]
                encoded = encoded * scale[:,None,None]
        
        return encoded.to(dtype=torch.float32), label_a.to(dtype=torch.float32), label_b.to(dtype=torch.float32)
    #input_ids, attention_mask, label_a, label_b
    
    def __len__(self):
        return self.len
    
    def get_attr_(self, index):
        """Since the subset object is garbage and 
        can't be used with my custom methods I'm doing this instead"""
        

    def random_split(self, val_size = 0.3):
        """
        split the dataset WRT patient repertoire!!, 
        ex val_size = 0.3 will take 0.3 of the total number of patients 
        and then query those patients and take the remaining 0.7 of other patients into train set
        """
        #Name indices
        train_idx = random.sample(range(len(self.unique)), math.floor((1-val_size)*len(self.unique)))
        train_names = self.unique[train_idx]
        
        idx = np.in1d(self.patients, train_names)
        # Returns the dataset objects instead of the Subset object so I can have easier access
        # to all my user-defined methods for this dataset class.
        # Splitting the train set takes about ~35 seconds
        train_subset = EmersonSubset(self, np.where(idx)[0])
        valid_subset = EmersonSubset(self, np.where(~idx)[0])
        return train_subset, valid_subset
    
    def get_patient(self, patient_filename):
        #If single patient (string)
        if type(patient_filename)==str:
            index = self.patients == patient_filename
            num_per_patient = [len(index==True)]
            label_idx = [0]
        #If multiple patients (list, np array, tuple, etc)
        else: 
            index = np.empty(0, dtype=np.int64)
            num_per_patient = []
            for p in patient_filename:
                idx = np.where(self.patients==p)[0]
                num_per_patient.append(len(idx==True))
                index = np.append(index,idx)
                
            # getting the index first element of each patient , 
            # Ex if num_per_patients is [9999, 10000, 10000]
            #label idx will be [0, 9999, 19999, 29999]
            # allowing us to access the first element of each labels
            label_idx = [0]
            for i, num in enumerate(num_per_patient[:-1]):
                label_idx.append(label_idx[i]+num_per_patient[i])
                
        encoded, label_a, label_b = self.__getitem__(index)
        return encoded, label_a[label_idx], label_b [label_idx], num_per_patient
    
    def random_sample(self, n):
        indices = random.sample(range(self.len), n)
        return self.__getitem__(indices)
    
    def get_random_patients(self, n):
        patients = random.sample(list(self.unique), n)

        return self.get_patient(patients)
    
#
class EmersonSubset(EmersonDataset):
    """
    Class inheritance to get an actual subset that can keep my methods because 
    torch.utils.data.Subset doesn't want to keep the user-defined methods lol
    """
    def __init__(self, emerson_dataset, index):
        self.seq =  emerson_dataset.seq[index]
        self.max_len = emerson_dataset.max_len
        self.rel_freq =  emerson_dataset.rel_freq[index]
        self.hla_a1 =  emerson_dataset.hla_a1[index]
        self.hla_a2 =  emerson_dataset.hla_a2[index]
        self.hla_b1 =  emerson_dataset.hla_b1[index]
        self.hla_b2 =  emerson_dataset.hla_b2[index]
        self.patients = emerson_dataset.patients[index]
        self.unique = np.unique(self.patients)
        self.len = len(self.seq)
        self.scale = emerson_dataset.scale
        