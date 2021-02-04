import pickle
import torch
import numpy as np
import sys,os,re,csv,pathlib

AAs = np.array(list('WFGAVILMPYSTNQCKRHDE')) #Constant list of amino acids
PAT = re.compile('[\\*_XB]')  ## non-productive CDR3 patterns

#Loading the pre-set Dictionary with values from PCA1-15 AA indices
with open('AAidx_dict.pkl', 'rb') as f: 
    AAidx_Dict = pickle.load(f) 

n_feats = len(AAidx_Dict['C']) # 15 features

#def standardize(train_data, test_data):
#    """
#    Sets the train data's mean to 1 and variance to 0
#    Applies the same operation to the test set
#    """

def one_hot_labels(target):
    tmp = target.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def shuffle_data(features, target, return_indices=False):
    """Shuffling the indices of a tensor"""
    indices = torch.randperm(features.size(0))
    data = features[indices]
    labels = target[indices]
    if return_indices:
        return data, labels, indices
    else: return data, labels

def naive_split(features, target, ratio):
    """
    naïvely split the datasets into train and validation
    """
    #Shuffling
    data, labels = shuffle_data(features, target)
    #Splitting
    z = math.ceil( (1-ratio)* labels.size(0) )
    train_data = data[0:z]
    train_labels = labels[0:z]
    eval_data = data[z:]
    eval_labels = labels[z:]
    return train_data, train_labels, eval_data, eval_labels


def read_seq(filename):
    """
    Read sequences from a txt, and returns an array of the sequences.
    Using numpy arrays because of easier fancy indexing when generating data from sequences
    """
    if '.txt' not in filename:
        print("Non .txt file given, exiting")
        return

    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            seq = line.strip()
            if not seq.startswith('C') or not seq.endswith('F'):continue
            data.append(seq)
    return np.array(data)

def aaindex_encoding(seq, device):
    """Encodes the AA indices to a given sequence"""
    n_aa = len(seq)
    temp = np.zeros([n_aa, n_feats], dtype=np.float32)
    for idx in range(n_aa):
        aa = seq[idx]
        temp[idx] = AAidx_Dict[aa]
    temp = np.transpose(temp)
    aa_encoding = torch.from_numpy(temp)
    aa_encoding = aa_encoding.unsqueeze(0)
    if device == torch.device('cuda'):
        aa_encoding = aa_encoding.to(device)
    return aa_encoding


def generate_features_labels(tumor_sequences, normal_sequences, device=None, shuffle=True):
    """For each CDR3 dataset (tumor and normal) sequences, get the feature vectors and labels"""
    
    #Normally, sequences are extracted as lists, but maybe I can modify something in read_sequences to return array instead of list
    if type(tumor_sequences)!=np.ndarray : tumor_sequences = np.array(tumor_sequences)
    if type(tumor_sequences)!=np.ndarray : normal_sequences = np.array(normal_sequences)

    #length of each datapoint (sequence)
    seqlens_tumor = np.array([len(seqs) for seqs in tumor_sequences]) 
    seqlens_normal = np.array([len(seqs) for seqs in normal_sequences])

    feature_dict, label_dict = {}, {}
    #Only keep sequences with length 12 to 16
    for length in range(12,17):
        #Using numpy to create mask for fancy indexing, converting to tensors later
        mask_tumor = np.where(seqlens_tumor==length)[0]
        mask_normal = np.where(seqlens_normal==length)[0]
        #Reusing the code from DeepCAT for Labels
        
        data = []
        for seqs in tumor_sequences[mask_tumor]:
            if len(PAT.findall(seqs))>0:continue #Skipping a sequence if it matches an unwanted CDR3 pattern 
            data.append(aaindex_encoding(seqs, device))
        nb_tumors = len(data)

        for seqs in normal_sequences[mask_normal]:
            if len(PAT.findall(seqs))>0:continue #Skipping a sequence if it matches an unwanted CDR3 pattern 
            data.append(aaindex_encoding(seqs, device))
        nb_normal = len(data[nb_tumors:])
        #Getting the labels 
        labels = torch.tensor(([1]*nb_tumors+[0]*nb_normal), dtype=torch.int64)
        data = torch.stack(data) #Stack a list of tensors into a single tensor

        #Shuffle the dataset by default because we simply added tumors followed by non tumors
        if shuffle:
            data, labels = shuffle_data(data, labels)

        #Sends to cuda. Shouldn't do this in batch-train because every tensors will be on GPU
        #leading to out of memory issues
        if device == torch.device('cuda'):
            #print(data.device)
            #data = data.to(device)
            labels = labels.to(device)

        feature_dict[length] = data
        label_dict[length] = labels 
        del data
        del labels
    print("Data device =",feature_dict[12].device)
    return feature_dict, label_dict

