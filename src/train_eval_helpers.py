from src.preprocessing import *
from src.models import deepcat_cnn
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.   data import BatchSampler, RandomSampler    
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import math
#from src.torch_util import save_checkpoint
from tqdm import tqdm


def train_model_step(model, criterion, optimizer, train_data, train_labels, mini_batch_size):
    """
    Trains one version (length) of a given model
    Trains over mini-batches for one epoch.
    Another function should call do the task of getting each of the 5 models and datasets
    and call this function once for each pair.
    """
    model.train()
    train_loss = 0
    #Minibatch SGD, get a list of indices to separate the train data into batches 
    for b in BatchSampler(RandomSampler(range(train_data.size(0))),
                          batch_size=mini_batch_size, drop_last=False):
        #Standard train loops
        output, _, _ = model(train_data[b])
        loss = criterion(output, train_labels[b]) #criterion = nn.CEL
        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= math.floor(len(train_labels)/mini_batch_size)
    #if e%200==0:
     #   print("Loss at epoch {e} : {l}".format(e=epoch,l=train_loss))
    return train_loss

def eval_model(model, criterion, data, labels, return_curve=False):
    """
    Takes a model & evaluates it (AUC & Accuracy)
    #TODO MAKE PLOTS
    """
    model.eval()
    one_hot = one_hot_labels(labels).cpu() # from preprocessing.py
    
    #Full test_set in case of unbalanced data
    #Logits = raw for loss, predictions = argmax(logit), probs = softmax(logit)
    logits, predictions, probs = model(data) 
    #predictions = logits.argmax(1)
    #probs = logits.softmax(1)
    
    eval_loss = criterion(logits, labels) #criterion = nn.CEL
    acc = accuracy_score(labels.cpu(), predictions.cpu())
    AUC = roc_auc_score(one_hot, probs.detach().cpu())
    f1 = f1_score(labels.cpu(), predictions.cpu(), labels=None, pos_label=1)
    
    if return_curve:
        curve = roc_curve(labels.detach().cpu(), probs.detach().cpu()[:,1], pos_label=1)
        return eval_loss.item(), acc, AUC, f1, curve
    
    else:return eval_loss.item(), acc, AUC, f1

def get_pred_df(model_dict, data_dict, target_labels_dict):
    """
    Evaluates each model on their targets, then returns a df containing 
    all the stats regarding the predictions.
    example of usage : 
    Use test sets as data_dict, target_labels_dict, load trained model into model_dict, 
    then call this method
    """
    RANGE = model_dict.keys()
    df = pd.DataFrame(columns = ['seqlen','y_true','predicted','prob_cancer',
                                 'tp','fp','tn','fn'])
    for ll in RANGE: 
        #seqlen = ll.value
        mod = model_dict[ll].to('cuda')
        X = data_dict[ll].to('cuda') #data
        y_true = target_labels_dict[ll] #y_true
        _, preds, probs = model_dict[ll](X)
        tmp_data = torch.cat((torch.full((len(preds),1), ll),#seqlen
                              y_true.view(-1,1).cpu(),#y_true
                              preds.detach().cpu().view(-1,1), #predicted
                              probs.detach().cpu()[:,1].view(-1,1)),
                             1)#cat dimension
        
        tmp = pd.DataFrame(data=tmp_data.numpy(),
                           columns =['seqlen','y_true','predicted','prob_cancer'])
        tmp['tp'] = tmp.apply(lambda x: 1 if (x['y_true']==x['predicted'] and x['predicted']==1) else 0, axis=1)
        tmp['fp'] = tmp.apply(lambda x: 1 if (x['y_true']!=x['predicted'] and x['predicted']==1) else 0, axis=1)
        tmp['tn'] = tmp.apply(lambda x: 1 if (x['y_true']==x['predicted'] and x['predicted']==0) else 0, axis=1)
        tmp['fn'] = tmp.apply(lambda x: 1 if (x['y_true']!=x['predicted'] and x['predicted']==0) else 0, axis=1)
        df = pd.concat([df,tmp], ignore_index=True)
    df=df.astype({'seqlen': 'int64', 'y_true':'int64','predicted':'int64',
                 'tp':'int64','fp':'int64','tn':'int64','fn':'int64'},copy=True)
    return df
#
def train_model_full(model, criterion, optimizer, nb_epochs, 
                     train_data, train_label, eval_data, eval_label, 
                     mini_batch_size, fname = '',
                     save = '', args = '', verbose=True):
    """
    Calls the method train_model_step & eval_model nb_epoch times.
    should do the split before calling !!!
    """
    epoch_print = math.floor(nb_epochs/3)#set an epoch to which we print
    train_losses = []
    val_losses = [] 
    val_accs = []
    val_aucs = []
    val_f1 = []

    if save == 'val':
        best_val == 1 #value to check for criterion = Val loss is compare if it is lower than 1 (initially)
    
    #elif save != 'val' and save != False:
    #    is_best == 0 #value to check for criterion = else, is compare if it is higher than 0 (initially)
        
                     
    for e in tqdm(range(nb_epochs)):
        train_loss = train_model_step(model, criterion, optimizer, train_data, train_label, mini_batch_size)
        train_losses.append(train_loss)
        val_loss, acc, auc, f1 = eval_model(model, criterion, eval_data, eval_label, roc_curve=False)
        val_losses.append(val_loss)
        val_accs.append(acc)
        val_aucs.append(auc)
        val_f1.append(f1)
        if (e%epoch_print==0 or e==nb_epochs)&verbose:
            print("Current stats at epoch = {} :\n"\
                  "Train loss = {}\n Val loss = {}"\
                  "\n Acc = {}\nAUC = {}\nF1 = {}".format(e,train_loss, val_loss,
                                                          acc, auc, f1))
        if save=='val':
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                torch.save({
                    'epoch':epoch,
                    'model':model.name,
                    'best_metric':save,
                    'state_dict': model.state_dict(),
                    'args':args,
                    'val_loss' : val_loss,
                    'acc' : acc,
                    'AUC' : auc,
                    'F1' : f1}, 
                    fname)
                     
            
    return train_losses, val_losses, val_accs, val_aucs, f1
                
def kfold_cv(model, criterion, optimizer, nb_epochs, kfold,
             mini_batch_size, data, labels, verbose=False):
    """Performs Kfold crossvalidation of a given model."""
    #These will be lists of lists
    kf = KFold(n_splits=kfold, shuffle=True)
    val_result = []
    acc_result = []
    AUC_result = []
    train_result = []
    f1_result = []
    epoch_print = math.floor(nb_epochs/3)#set an epoch to which we print
    
    for iteration, (train_index, eval_index) in enumerate(kf.split(data)):
        print("Crossvalidating : Fold = ", iteration+1)
        model.reset_parameters()
        train_data, train_labels = data[train_index], labels[train_index]
        eval_data, eval_labels = data[eval_index], labels[eval_index]
        train_temp = []
        val_temp = []
        acc_temp = []
        AUC_temp = []
        f1_temp = []
        #print("Starting training")
        for e in tqdm(range(nb_epochs)):
            train_loss = train_model_step(model, criterion, optimizer, train_data, train_labels, mini_batch_size)
            val_loss, acc, AUC, f1 = eval_model(model, criterion, eval_data, eval_labels)
            train_temp.append(train_loss)
            val_temp.append(val_loss)            
            acc_temp.append(acc)            
            AUC_temp.append(AUC)
            f1_temp.append(f1)
            if (e%epoch_print==0|e==nb_epochs)&verbose:
                print("Current stats at epoch = {} :\n"\
                      "Train loss = {}\n Val loss = {}"\
                      "\nAcc = {}\nAUC = {}".format(e,train_loss,
                                                     val_loss,
                                                     acc, AUC))
        train_result.append(train_temp)    
        val_result.append(val_temp)
        acc_result.append(acc_temp)
        AUC_result.append(AUC_temp)
        f1_result.append(f1_temp)

    return train_result, val_result, acc_result, AUC_result, f1_result

#def batch_run(function, *args):
#    for lengths in range(12,17):function(length,*args)