"""
CORRESPONDS TO A BETTER VERSION OF THEIR SCRIPT PREPAREADAPTIVEFILE.PY
This one is written in pandas and is both more efficient and much cleaner.
"""

import os
from os.path import exists
import numpy as np
import csv
from csv import reader
import sys
import pandas as pd
import argparse

#The fields and mapping used by the base script
FIELDS = ['aminoAcid','vMaxResolved', 'frequencyCount (%)']
MAPPING = {k:v for (k,v) in zip(FIELDS, ['amino_acid', 'v_gene', 'frequency'])}

def clean_tsv(tsv_file, save_filename, threshold=10000):
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
	try:
		tmp = pd.read_csv(tsv_file, sep='\t',usecols=FIELDS)[FIELDS] #read only used columns
	except ValueError:
		print("Couldn't read {file}, please check that the columns header contains {FIELD}".format(file=tsv_file,FIELD=FIELDS))

	print("Currently reading : ", tsv_file)
	tmp=tmp.query('vMaxResolved!="unresolved"') #dropping unresolved
	#Check if length motifs within [10,24]
	len_mask = (tmp['aminoAcid'].str.len() >= 10) &(tmp['aminoAcid'].str.len() <= 24)
	
	#Check if starts with C and ends with F
	motif_mask = tmp['aminoAcid'].str.startswith('C', na=False) &\
	             tmp['aminoAcid'].str.endswith('F', na=False) 
	
	#Check if sequence contains any these patterns
	patterns = '|'.join(['\*','X'])
	contains_mask = ~(tmp['aminoAcid'].str.contains(patterns, na= False))

	mask = len_mask & motif_mask & contains_mask
	tmp = tmp[mask].sort_values('frequencyCount (%)', ascending=False)
	tmp = tmp.rename(columns=MAPPING)
	if threshold is None:
		tmp.to_csv(save_filename, sep='\t', index = False)
		print("File saved under ",save_filename)
		return
	else:
		tmp=tmp.iloc[:threshold] #note : there will be threshold+1 rows due to the header.
		tmp.to_csv(save_filename, sep='\t', index = False)
		print("File saved under ",save_filename)
		return

def args_parser():
	parser = argparse.ArgumentParser(description='Processes a raw .tsv file TCR sequences.')
	parser.add_argument('-indir', type = str, default = os.getcwd(), help = 'Input directory containing the raw .tsv files of interest. By default, it is the current working directory')
	parser.add_argument('-thr',type = int, default = 10000, help = 'Number of lines to keep. By default, take 10 000 entries.')
	return parser.parse_args()

def main():
	"""
	Lists all .tsv files in the input directory, and applies clean_tsv() to them.
	Then saves them in the output directory called 'tsv_output/' located within the input directory
	^ subject to changes depending on the full pipeline requirements later on^
	"""
	args = args_parser()
	input_directory = args.indir
	if not input_directory.endswith('/'):
		input_directory = input_directory+'/'
		print("HERE", input_directory)
	output_directory = input_directory+'tsv_output/'

	threshold = args.thr
	#Listing .tsvs, 
	filenames = os.listdir(input_directory)
	for filename in filenames:
		if ('.tsv' not in filename):continue #Skipping non .tsv file
		if not os.path.exists(output_directory):
			os.makedirs(output_directory)   

		save_filename = output_directory+'TestReal-'+filename

		clean_tsv(input_directory+filename, save_filename, threshold)


if __name__ == '__main__':
	main()
