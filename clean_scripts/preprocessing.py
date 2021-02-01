import pickle
import torch
import numpy as np
import sys,os,re,csv,pathlib

AAs = np.array(list('WFGAVILMPYSTNQCKRHDE')) #Constant list of amino acids
PAT = re.compile('[\\*_XB]')  ## non-productive CDR3 patterns

#Loading the pre-set Dictionary with values from PCA1-15 AA indices
with open('./AAidx_dict.pkl', 'rb') as f: 
    AAidx_Dict = pickle.load(f) 

n_feats = len(AAidx_Dict['C']) # 15 features


def onehot_encoding(seq):
	"""One-hot encodes a given sequence"""

	n_aa = len(seq) #number of amino acids in seq
	#Re-using their notation and converting to torch.Tensor
	temp = np.zeros([20, n_aa], dtype = np.float32) #Would've used int but keeping this for now
	for idx in range(n_aa):
		aa = seq[idx]
		mask = np.where(AAs==aa)
		temp[mask, idx]=1
	onehot = torch.from_numpy(temp)
	return onehot

def aaindex_encoding(seq):
	"""Encodes the AA indices to a given sequence"""
	n_aa = len(seq)
	temp = np.zeros([n_aa, n_feats], dtype=np.float32)
	for idx in range(n_aa):
		aa = seq[idx]
		temp[idx] = AAidx_Dict[aa]
	temp = np.transpose(temp)
	aa_encoding = torch.from_numpy(temp)
	return aa_encoding


def get_feature_labels(tumor_data, normal_data):
	"""For each CDR3 dataset (tumor and normal), get the feature vectors and labels"""
	#number of datapoints(sequences)
	n_tumor = len(tumor_data) 
	n_normal = len(normal_data)
	#length of each datapoint (sequence)
	seqlens_tumor = np.array([len(seqs) for seqs in tumor_data]) 
	seqlens_normal = np.array([len(seqs) for seqs in normal_data])

	feature_dict, label_dict = {}, {}

	for length in range(12,17):
		mask_tumor = np.where(seqlens_tumor==length)[0]
		mask_normal = np.where(seqlens_normal==length)[0]
		#Reusing the code from DeepCAT for Labels
		labels = np.array([1]*len(mask_tumor)+[0]*len(mask_normal), dtype=np.int32)
		data = []

		for seqs in tumor_data[mask_tumor]:
			if len(PAT.findall(seqs))>0:continue #Skipping a sequence if it matches an unwanted CDR3 pattern 
			data.append(aaindex_encoding(seqs))

		for seqs in normal_data[mask_normal]:
			if len(PAT.findall(seqs))>0:continue #Skipping a sequence if it matches an unwanted CDR3 pattern 
			data.append(aaindex_encoding(seqs))
		data = np.array(data)
		features = {'x':data, 'length':length}
		feature_dict[length] = features
		label_dict[length] = labels 

	return feature_dict, label_dict
