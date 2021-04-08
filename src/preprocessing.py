import pickle
import torch
import numpy as np
import sys,os,re,csv,pathlib, math 
import pandas as pd 
import warnings
from src.pickling import load_all, load_pkl
warnings.filterwarnings("ignore", category=DeprecationWarning) 

AAs = np.array(list('WFGAVILMPYSTNQCKRHDE')) #Constant list of amino acids
PAT = re.compile('[\\*_XB]')  ## non-productive CDR3 patterns

#Loading the pre-set Dictionary with values from PCA1-15 AA indices
PATH = os.getcwd()
#Merged dict : [atchley1 ... atchley5, PCA1,...,PCA15]
AAidx_Dict, merged_dict , minmax_aaidx, minmax_merged, minmax_atchley,  hla_a,  hla_b = load_all(PATH) #See src.pickling.py
AA_KEYS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
CHAR_TO_INT = dict((c,i) for i,c in enumerate(AA_KEYS))
INT_TO_CHAR = dict((i,c) for i,c in enumerate(AA_KEYS))
    
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
    Pre-saving to .txt is much faster than reading from .tsv files
    Using numpy arrays because of easier fancy indexing when generating data from sequences
    """
    if '.txt' not in filename:
        print("Non .txt file given, exiting")
        return

    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            seq = line.strip()
            #if not seq.startswith('C') or not seq.endswith('F'):continue
            data.append(seq)
    return np.array(data)

def onehot_aa(seq, pad_sequence = True, max_len = 23):
    "encodes a single sequence to one-hot "
    int_encoded = [CHAR_TO_INT[char] for char in seq]
    onehot_encoded = list()
    for value in int_encoded:
        letter = [0 for _ in range(len(AA_KEYS))]
        letter[value] = 1
        onehot_encoded.append(letter)
    #onehot_encoded = 
    return torch.tensor(onehot_encoded)
    #if pad_sequence==True:
    #    zeros=torch.zeros(((max_len-onehot_encoded.shape[0]), 20))
    #    return torch.cat((onehot_encoded, zeros))
    #
    #else :
    #    return onehot_encoded

def pad_seq(seq, max_len=23):
    "Takes a one-hot encoded sequences and pads it to max_len"
    zeros = torch.zeros(((max_len-seq.shape[0]), 20))
    return torch.cat((seq, zeros))

def onehot_batch(sequences, max_len = 23):
    """takes a bunch of sequences and onehot encode +pad them"""
    # unsqueeze here adds an extra dimension ("channel")
    # in case we want to treat the batch as an image
    return torch.stack([pad_seq(onehot_aa(x)) for x in sequences])#.unsqueeze(1) 

def onehot_decode(onehot):
    return INT_TO_CHAR[argmax(onehot[0])]    
    
def aaindex_encoding(seq, device=None, scaling ='minmax'):
    """Reads a string (seq), Encodes the AA index PCA values to a given sequence"""
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
    """Reads a string (seq), Encodes the AA indices to a given sequence with atchley+aaidxPCA factors merged"""
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

def atchley_encoding(seq, device=None, scaling='minmax'):
    """Reads a string (seq), Encodes the AA indices to a given sequence with atchley+aaidxPCA factors merged"""
    n_aa = len(seq)
    temp = np.zeros([n_aa, 5], dtype=np.float32)
    for idx in range(n_aa):
        aa = seq[idx]
        temp[idx]=minmax_atchley[aa]
    temp = np.transpose(temp)
    aa_encoding = torch.from_numpy(temp)
    aa_encoding = aa_encoding.unsqueeze(0)
    if device == torch.device('cuda'):
        aa_encoding = aa_encoding.to(device)
    return aa_encoding

def generate_features_labels(tumor_sequences, normal_sequences, keys = range(12,17), 
                             device=None, shuffle=True, encoding = 'aaidx',
                             scaling='minmax', crop = False):
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
            if crop == True:
                seqs= seqs[3:-3]
            if encoding == 'aaidx': 
                data.append(aaindex_encoding(seqs, device, scaling))
            elif encoding == 'aa_atchley':
                data.append(aaidx_atchley_encoding(seqs, device, scaling))
            elif encoding == 'atchley':   
                data.append(atchley_encoding(seqs, device, scaling))
                        
        nb_tumors = len(data)

        for seqs in normal_sequences[mask_normal]:
            if len(PAT.findall(seqs))>0:continue #Skipping a sequence if it matches an unwanted CDR3 pattern 
            if crop == True:
                seqs= seqs[3:-3]
            if encoding == 'aaidx': 
                data.append(aaindex_encoding(seqs, device, scaling))
            elif encoding == 'aa_atchley':
                data.append(aaidx_atchley_encoding(seqs, device, scaling))
            elif encoding == 'atchley':
                data.append(atchley_encoding(seqs, device, scaling))
                
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
            labels = labels.to(device)

        feature_dict[length] = data
        label_dict[length] = labels 
        del data
        del labels
    print("Data device =",feature_dict[12].device)
    print("Done loading, returning features and labels.")
    return feature_dict, label_dict

def get_train_test_data(directory, keys, device=None, shuffle = True, 
                        encoding = 'aaidx', scaling ='minmax', crop = False):
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
            print("Reading : ", f)
            if 'tumor' in lower:
                test_tumor = read_seq(directory+f)
            elif 'normal' in lower:
                test_normal = read_seq(directory+f)
        else:# 'test' not in lower:
            print("Reading : ", f)
            if 'tumor' in lower:
                train_tumor = read_seq(directory+f)
            elif 'normal' in lower:
                train_normal = read_seq(directory+f)
    print("\nTrain")
    train_feats_dict, train_labels_dict = generate_features_labels(train_tumor, train_normal, keys, device, shuffle, encoding, scaling, crop)
    print("\nTest")
    test_feats_dict, test_labels_dict = generate_features_labels(test_tumor, test_normal, keys, device, shuffle, encoding, scaling, crop)
    
    return train_feats_dict, train_labels_dict, test_feats_dict, test_labels_dict



def read_adaptive_tsv(tsv_file, save=False, threshold=10000, savedir=None):
    """
    Reads a raw .tsv containing information related to TCR sequences
    Cleans them and retain the following informations : 
        columns : amino_acids, v_gene (vMaxResolved), frequency (frequencyCount (%))
        rows : only rows that have 
                - defined sequences (doesn't contain X or *)
                - sequence length >=10 <=24
                - sequence starts with C and ends with F
                - does not contain "unresolved" in vMaxResolved
    """
    no_resolve = False
    try:
        FIELDS = ['amino_acid', 'v_gene','v_resolved', 'frequency']
        DTYPES={'amino_acid':str, 'v_gene' :str, 'v_resolved':str, 'frequency':float}
        tmp = pd.read_csv(tsv_file, sep='\t',usecols=FIELDS, dtype=DTYPES)[FIELDS] #read only used columns
    except ValueError:
        try:
            FIELDS = ['amino_acid', 'v_gene', 'frequency']
            DTYPES={'amino_acid':str, 'v_gene' :str, 'frequency':float}
            tmp = pd.read_csv(tsv_file, sep='\t',usecols=FIELDS, dtype=DTYPES)[FIELDS] #read only used columns
            no_resolve = True
        except:
            print("Couldn't read {file}, please check that the columns header contains {FIELD}".format(file=tsv_file,FIELD=FIELDS))
            return

    print("Currently reading : ", tsv_file, end='\r')
    tmp=tmp.query('v_gene!="unresolved"') #dropping unresolved
    tmp = tmp.dropna(subset=['amino_acid'])
    #print("\n\n\n######################",tmp,"\n\n\n#######################")
    tmp['len'] = tmp.apply(lambda x: len(x['amino_acid']),axis=1).copy() #Getting the length of a sequence
    #Check if length motifs within [10,24]
    len_mask = (tmp['len'] >= 10) & (tmp['len'] <= 24)
    
    #Check if starts with C and ends with F
    motif_mask = tmp['amino_acid'].str.startswith('C', na=False) &\
                 tmp['amino_acid'].str.endswith('F', na=False) 
    
    #Check if sequence contains any these patterns
    patterns = '|'.join(['\*','X'])
    contains_mask = ~(tmp['amino_acid'].str.contains(patterns, na= False))
    #Combined the 3 masks
    mask = len_mask & motif_mask & contains_mask 
    tmp = tmp[mask].sort_values('frequency', ascending=False)

    if threshold is None:
        if save==True:    
            save_filename = tsv_file.split('.tsv')[0]+'_parsed.txt'
            save_filename = os.path.join(savedir, save_filename)
            tmp.to_csv(save_filename, sep='\t', index = False)
            print("File saved under ",save_filename, end='\r')

    else:
        tmp=tmp.iloc[:threshold] 
        if save==True:
            save_filename = tsv_file.split('.tsv')[0]+'_parsed.txt'
            save_filename = os.path.join(savedir, os.path.basename(save_filename))
            tmp.to_csv(save_filename, sep='\t', index = False)
            print("File saved under ",save_filename, end='\r')
            
        if no_resolve==True:
            return tmp[['amino_acid','v_gene','frequency']]
        else : 
            return tmp[['amino_acid','v_resolved','frequency']]
    
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
        elif encoding == 'atchley':
            feats.append(atchley_encoding(seq,device,scaling))

    feats = torch.stack(feats).to(device)
    return feats


def get_ab_label(a1, a2, b1, b2):
    """
    Takes the known HLA a1, a2, b1, b2 alleles (str)
    Computes the A and B labels as 1hot encoded vector.
    
    For example, if a patient has allele 'A11', 'A03', for HLA-A,
    Given hla_a = np.array(['A01', 'A02', 'A03', 'A11', 'A23', 'A24', 'A25', 'A26', 'A28', 'A29',
    'A30', 'A31', 'A32', 'A33', 'A34', 'A36', 'A66', 'A68', 'A69', 'A74', 'A80'])
    his A_label will be [0, 0, 1, 1, 0, ..., 0]
    """
    
    A_label = torch.from_numpy(((hla_a==a1) | (hla_a == a2)).astype(int))
    B_label = torch.from_numpy(((hla_b==b1) | (hla_b == b2)).astype(int))
    return A_label, B_label

def get_all_label(a1_array, a2_array, b1_array, b2_array):
    #If somehow a single element (i.e. a string) was passed to get_all_label, do this
    if type(a1_array) == str:
        a,b = get_ab_label(a1_array, a2_array, b1_array, b2_array)
    #Else, if it's an array, list, tuple, whatever iterable, do this instead
    else:
        labs = [get_ab_label(x1,x2,y1,y2) for (x1,x2,y1,y2) in zip(a1_array,a2_array,b1_array,b2_array)]
        a = torch.stack([x[0] for x in labs]).double()
        b = torch.stack([x[1] for x in labs]).double()
    return a, b
    
def get_allele_name(a_label, b_label):
    """reverse mapping from onehot to allele names"""
    return hla_a[a_label.bool()], hla_b[b_label.bool()]