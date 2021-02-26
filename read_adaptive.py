import os
from os.path import exists
import numpy as np
import csv
from csv import reader
import sys
import pandas as pd
import argparse
from src.preprocessing import *
from src.ismart import * 
from datetime import datetime as dt 

def args_parser():
    parser = argparse.ArgumentParser(description='processes folders with subfolders containing .tsv files so that DeepCAT garbage program can read it')
    parser.add_argument('-indir', type = str, help = 'Input directory containing the raw .tsv files of interest. By default, it is the current working directory')
    parser.add_argument('-thr',type = int, default = 10000, help = 'Number of lines to keep. By default, take 10 000 entries.')
    return parser.parse_args()

def main():
    start_time = dt.now()
    #Getting input dir
    args = args_parser()
    input_directory = args.indir
    if not input_directory.endswith('/'):
        input_directory = input_directory+'/'
    #Getting output dir which is inputdir + parsed/
    output_directory = os.path.join(input_directory, 'parsed/')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok = True) 

    threshold = args.thr
    #Listing .tsvs, 
    for subfolder in os.listdir(args.indir): #For each folder in in_directory
        if 'parsed' in subfolder:continue
        subfolderpath = os.path.join(args.indir, subfolder)#Get path to subfolder
        #Getting save directory
        sub_base = os.path.basename(subfolderpath)+'_parsed/'
        save_dir = os.path.join(output_directory, sub_base)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok = True)    

        #Listing the .txt files that are in this folder
        files = [os.path.join(subfolderpath,x) for x in os.listdir(subfolderpath) if x.endswith('.tsv')]
        scores = []
        #Iterating over each file and getting the scores for each file
    for filename in files:
        fn=os.path.basename(filename)
        df = read_adaptive_tsv(filename, save=False, threshold = args.thr)
        df = ismart(df, save = False)   
        df.rename(columns = {'amino_acid':'aminoAcid'}, inplace=True)
        df.to_csv(os.path.join(save_dir,fn.split('.tsv')[0]+'_parsed_clusteredCDR3.txt'),
            sep = '\t', header=True, index=False)
    end_time = dt.now()       
    elapsed = divmod((end_time-start_time).total_seconds(), 60)
    print("\nTime elapsed:\n\t{} minutes\n\t{} seconds".format(elapsed[0], elapsed[1]))

if __name__ == '__main__':
    main()
