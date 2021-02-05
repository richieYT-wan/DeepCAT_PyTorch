"""
trains the 5 networks (or can choose which length to train [or not])

"""
#Allows relative imports, depends on where this script will end-up
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import pickle 

from src.models import *
from src.preprocessing import *
from src.torch_util import *
from src.train_eval_helpers import eval_model, train_model_full, kfold_cv
#from src.plots import *

import argparse
from src.torch_util import str_to_bool
from datetime import datetime as dt 

def args_parser():
    parser = argparse.ArgumentParser(description='Trains the model. Can use naïve splitting or K-fold crossvalidation.')
    parser.add_argument('-indir', type = str, default = './TrainingData/', help = 'relative path to input directory containing the tumor/normal, train&test CDR3 sequences. Format should be .txt, with no header. By default, it is ./TrainingData/, assuming train.py is located in the root of the github folder (./DeepTCR_PyTorch/TrainingData/ with ./DeepTCR_PyTorch/train.py.')
    parser.add_argument('-outdir', type = str, default= 'output/', help = 'Relative path to the output directory where the best weights as well as figures, training losses/accuracies etc. are logged. By default, if it does not exist, a folder called output/ is created in the current directory.')
    parser.add_argument('-nb_epochs', type=int, default = 300, help = 'Number of epochs over which the model is trained.')
    parser.add_argument('-lr', type = float, default = 0.00125, help= 'Learning rate, 0.00125 by default')
    parser.add_argument('-keys', nargs = "+", default = [12,13,14,15,16], help = 'A list of the models (lengths) to train (corresponds to the sequence length). By default, all models (len = 12 to 16) are trained. If a single model is needed, please input a list. ex : [12] to only train model 12')
    parser.add_argument('-batchsize', type=int, default=250, help ='Mini-batch size for training. By default, 250')
    parser.add_argument('-valmode', type= str, default='naive', help = 'Validation mode. By default, it is a "naïve" split of the training set with 0.67 as training, 0.33 as validation. Value should only be NAIVE or KCV. (not case sensitive)')
    parser.add_argument('-kfold', type = int, default = 5, help = 'If --valmode = KCV, then --kfold specifies K, i.e. the number of folds to crossvalidate over.')
    parser.add_argument('-ratio', type=float, default=1/3, help='the proportion of data used as validation set. By default, it is 0.33')
    parser.add_argument('-test', type=str_to_bool, default = True, help='Whether to include a test set to compute evaluation of the best model obtained during training. If False, the validation set will be used to compute the final evaluation. If val-mode is KCV, then --test should be True!!')
    parser.add_argument('-v', type=str_to_bool, default=True, help="Whether to print progress. (True/False), True by default")
    parser.add_argument('-metric', default="val", help = 'Which metric to use to log the best weights. Takes values in [val, acc, auc, f1]. By default, it is val.')
    #parser.add_argument('-loss', type = torch.)
    #parser.add_argument('')
    #parser.add_argument('--opt', dest=optim, type=torch.optim)
    return parser.parse_args()

def main():
    start_time = dt.now()
    args = args_parser()
    #for arg in args.argument():
    print("\nARGS:",args,"\n")
    #Reading data from train dir 
    TRAINDIR = args.indir
    KEYS = [int(k) for k in args.keys]
    OUTDIR = os.path.join(os.getcwd(), args.outdir) 
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR, exist_ok = True)  
    #If single learning rate given, expand to match number of keys (default behaviour)
    #Training hyperparameters
    print("\nOUTPUT DIRECTORY:",OUTDIR)
    lr = args.lr
    #if len(lr)==1: lr*=len(KEYS)
    nb_epochs = args.nb_epochs
    #if len(nb_epochs)==1: nb_epochs*=len(KEYS)
    mbs = args.batchsize #mini-batch_size
    
    #returns dictionaries!! e.g. train_feats[12] returns the feats for L=12
    if args.test==True:
        train_feats, train_labels, test_feats, test_labels = get_train_test_data(TRAINDIR, KEYS, device=None, shuffle=True) 
    elif args.test==False:
        train_feats, train_labels, _, _ = get_train_test_data(TRAINDIR, KEYS, device=None, shuffle=True) 
    #Set device to None. We don't want to send every tensor to 'cuda' as it will run out of memory. 
    #Instead, the tensors should be sent to Cuda only when needed.
    #Ex when L = 12, all the L = 12 tensors are sent to cuda, the rest stay on 'cpu'.
    #This extra copy may slow down the code but at least won't break due to GPU memory limitations.
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print("Using : {}".format(DEVICE))
    else:
        DEVICE = torch.device('cpu')
        print("Using : {}".format(DEVICE))

    #Non-robust implementation of this checking for now
    crossvalidate = args.valmode.lower() == 'kcv'
    naive = args.valmode.lower() == 'naive'
    
    if crossvalidate:
        kfold = args.kfold
        print("### ============ Using {}-Fold Crossvalidation ============ ###".format(kfold))
    elif naive:
        ratio = args.ratio
        print("### ============ Using naïve-split. Ratio = {} ============ ###".format(ratio))

    #Getting the models using get_models from src.models
    model_dict = get_models(KEYS)
    train_loss_dict = {}
    val_loss_dict = {}
    val_accs_dict = {}
    val_aucs_dict = {}
    val_f1_dict = {}
    
    for index, key in enumerate(KEYS):
        print("\n### ============ Starting training for model with sequence length = {} ============ ###\n".format(key))
        ID = str(key)
        data_temp = train_feats[key].detach().clone().to(DEVICE)
        labels_temp = train_labels[key].detach().clone().to(DEVICE)
        model_dict[key].to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_dict[key].parameters(), lr=lr)
        #Stuff for saving checkpoint
        filename = model_dict[key].name+'_best_'+args.metric+'.pth.tar'
        FNAME = os.path.join(OUTDIR, filename)
        
        if naive:
            train_data, train_target, eval_data, eval_target = naive_split(data_temp, labels_temp, ratio)
            train_losses, val_losses, val_accs, val_aucs, val_f1 = train_model_full(model_dict[key], criterion, 
                                                                            optimizer, nb_epochs,
                                                                            train_data, train_target, 
                                                                            eval_data, eval_target, 
                                                                            mbs, fname=FNAME,
                                                                            save=args.metric,args=args,verbose=True)    
        elif crossvalidate:
            train_losses, val_losses, val_accs, val_aucs, val_f1 = kfold_cv(model_dict[key], criterion, optimizer, nb_epochs, 
                                                                            kfold, mbs, data_temp, labels_temp, verbose=args.v)
            
            ratio = 1/kfold
            train_data, train_target, eval_data, eval_target = naive_split(data_temp, labels_temp, ratio)
            #After crossvalidation, re-run a full training and saves the weights (Very inefficient for now)
            model_dict[key].reset_parameters()
            _, _, _, _, _ = train_model_full(model_dict[key], criterion, optimizer, nb_epochs,
                                          train_data, train_target, eval_data, eval_target, 
                                          mbs, fname=FNAME, save=args.metric,args=args,verbose=True)
        
        train_loss_dict[key] =  train_losses
        val_loss_dict[key] = val_losses
        val_accs_dict[key] = val_accs
        val_aucs_dict[key] = val_aucs    
        val_f1_dict[key] = val_f1
        #Clearing GPU memory after it's done for the next model[key]
        model_dict[key].to('cpu') 
        del data_temp
        del labels_temp
    print("\n### ============ Training complete. Saving files. ============ ###\n")
    end_time = dt.now()
    elapsed = divmod((end_time-start_time).total_seconds(), 60)
    print("\nTime elapsed:\n\t{} minutes\n\t{} seconds".format(elapsed[0], elapsed[1]))

    fns=['train_losses_dict.pkl','val_losses_dict.pkl', 'val_accs_dict.pkl','val_aucs_dict.pkl','val_f1_dict.pkl']
    for index, item in enumerate([train_loss_dict, val_loss_dict, val_accs_dict, val_aucs_dict, val_f1_dict]):
        picklename = os.path.join(OUTDIR, fns[index])
        with open(picklename, 'wb') as f:
            pickle.dump(item, f)
    
    
if __name__ == '__main__':
    main()
