import pickle
import torch
import numpy as np
import sys,os,re,csv,pathlib, math 
import pandas as pd 

AAs = np.array(list('WFGAVILMPYSTNQCKRHDE')) #Constant list of amino acids
PAT = re.compile('[\\*_XB]')  ## non-productive CDR3 patterns

#Loading the pre-set Dictionary with values from PCA1-15 AA indices
PATH = os.getcwd()
#Merged dict : [atchley1 ... atchley5, PCA1,...,PCA15]
if 'notebook' in PATH:
    with open('../src/AAidx_dict.pkl', 'rb') as f: 
        AAidx_Dict = pickle.load(f) 
    with open('../src/merged_dict.pkl', 'rb') as g: 
        merged_dict = pickle.load(g)      
    with open('../src/minmax_aaidx.pkl','rb') as h:
        minmax_aaidx = pickle.load(h)
    with open('../src/minmax_merged.pkl','rb') as i:
        minmax_merged = pickle.load(i)

else :
    with open('./src/AAidx_dict.pkl', 'rb') as f: 
        AAidx_Dict = pickle.load(f) 
    with open('./src/merged_dict.pkl', 'rb') as g: 
        merged_dict = pickle.load(g) 
    with open('./src/minmax_aaidx.pkl','rb') as h:
        minmax_aaidx = pickle.load(h)
    with open('./src/minmax_merged.pkl','rb') as i:
        minmax_merged = pickle.load(i)


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
    ratio determines the proportion of data in the validation set.
    For example, ratio = 0.33 puts 33% of the data into the validation set.
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


def aaindex_encoding(seq, device, scaling ='minmax'):
    """Encodes the AA indices to a given sequence"""
    n_aa = len(seq)
    temp = np.zeros([n_aa, 15], dtype=np.float32)
    for idx in range(n_aa):
        aa = seq[idx]
        if scaling == 'minmax': 
            temp[idx]=minmax_aaidx[aa]
        else:
            temp[idx] = AAidx_Dict[aa]
    temp = np.transpose(temp)
    aa_encoding = torch.from_numpy(temp)
    aa_encoding = aa_encoding.unsqueeze(0)
    if device == torch.device('cuda'):
        aa_encoding = aa_encoding.to(device)
    return aa_encoding

def aaidx_atchley_encoding(seq, device=None, scaling ='minmax'):
    """Encodes the AA indices to a given sequence with atchley+aaidxPCA factors merged"""
    n_aa = len(seq)
    temp = np.zeros([n_aa, 20], dtype=np.float32)
    for idx in range(n_aa):
        aa = seq[idx]
        if scaling == 'minmax': 
            temp[idx]=minmax_merged[aa]
        else:
            temp[idx] = merged_dict[aa]
    temp = np.transpose(temp)
    aa_encoding = torch.from_numpy(temp)
    aa_encoding = aa_encoding.unsqueeze(0)
    if device == torch.device('cuda'):
        aa_encoding = aa_encoding.to(device)
    return aa_encoding

def generate_features_labels(tumor_sequences, normal_sequences, keys = range(12,17), 
                             device=None, shuffle=True, encoding = 'aaidx', scaling='minmax'):
    """For each CDR3 dataset (tumor and normal) sequences, get the feature vectors and labels"""
    
    #Normally, sequences are extracted as lists, but maybe I can modify something in read_sequences to return array instead of list
    if type(tumor_sequences)!=np.ndarray : tumor_sequences = np.array(tumor_sequences)
    if type(normal_sequences)!=np.ndarray : normal_sequences = np.array(normal_sequences)

    #length of each datapoint (sequence)
    seqlens_tumor = np.array([len(seqs) for seqs in tumor_sequences]) 
    seqlens_normal = np.array([len(seqs) for seqs in normal_sequences])
    print("Getting data")
    feature_dict, label_dict = {}, {}
    #Only keep sequences with length 12 to 16
    for length in keys:
        #Using numpy to create mask for fancy indexing, converting to tensors later
        
        mask_tumor = np.where(seqlens_tumor==length)[0]
        mask_normal = np.where(seqlens_normal==length)[0]
        #Reusing the code from DeepCAT for Labels
        
        data = []
        for seqs in tumor_sequences[mask_tumor]:
            if len(PAT.findall(seqs))>0:continue #Skipping a sequence if it matches an unwanted CDR3 pattern 
            if encoding == 'aaidx': 
                data.append(aaindex_encoding(seqs, device, scaling))
            elif encoding == 'aa_atchley':
                data.append(aaidx_atchley_encoding(seqs, device, scaling))
        nb_tumors = len(data)

        for seqs in normal_sequences[mask_normal]:
            if len(PAT.findall(seqs))>0:continue #Skipping a sequence if it matches an unwanted CDR3 pattern 
            if encoding == 'aaidx': 
                data.append(aaindex_encoding(seqs, device, scaling))
            elif encoding == 'aa_atchley':
                data.append(aaidx_atchley_encoding(seqs, device, scaling))
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
    print("Done loading, returning features and labels.")
    return feature_dict, label_dict

def get_train_test_data(directory, keys, device=None, shuffle = True, 
                        encoding = 'aaidx', scaling ='minmax'):
    """
    From a directory, reads the .txt and generates the corresponding train/test tumor-normal data (features+label). Assumes the files are named as 'NormalCDR3.txt', 'NormalCDR3_test.txt', 'TumorCDR3.txt', 'TumorCDR3_test.txt'
    """
    if not directory.endswith('/'):
        directory=directory+'/'
    files = os.listdir(directory)
    #Not pretty loops lol
    for f in files:
        lower = f.lower()
        if '.txt' not in lower:continue #Skips non .txt files
        if 'cdr3' not in lower:continue #Skips files without CDR3 in the name
        if 'test' in lower:
            if 'tumor' in lower:
                test_tumor = read_seq(directory+f)
            elif 'normal' in lower:
                test_normal = read_seq(directory+f)
        else:# 'test' not in lower:
            if 'tumor' in lower:
                train_tumor = read_seq(directory+f)
            elif 'normal' in lower:
                train_normal = read_seq(directory+f)
    print("\nTrain")
    train_feats_dict, train_labels_dict = generate_features_labels(train_tumor, train_normal, keys, device, shuffle, encoding, scaling)
    print("\nTest")
    test_feats_dict, test_labels_dict = generate_features_labels(test_tumor, test_normal, keys, device, shuffle, encoding, scaling)
    
    return train_feats_dict, train_labels_dict, test_feats_dict, test_labels_dict
            
def read_ismart(filename):
    """
    Reads a .txt file, assumes that it is in the format outputed by iSMARTm.py
    Returns a dataframe, whose underlying arrays are to be used for aaindex_encoding()
    """
    df = pd.read_csv(filename, sep='\t', header=0)
    df['len'] = df.apply(lambda x: len(x['aminoAcid']),axis=1)
    df = df.query('len>=12 & len<=16').copy()
    mask = df['aminoAcid'].str.startswith('C') & df['aminoAcid'].str.endswith('F')
    df = df[mask].sort_values('len', ascending=True).copy()
    #seqs = df['aminoAcid'].copy().values
    #return seqs, df
    return df

def get_feats_tensor(sequences, device, encoding='aaidx', scaling = 'minmax'):
    "Sequences : numpy array"
    feats = []
    for seq in sequences:
        if encoding == 'aaidx': 
            feats.append(aaindex_encoding(seq, device, scaling))
        elif encoding == 'aa_atchley':
            feats.append(aaidx_atchley_encoding(seq, device, scaling))
    feats = torch.stack(feats).to(device)
    return feats