from src.preprocessing import *
from src.models import deepcat_cnn
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.   data import BatchSampler, RandomSampler    
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import math
#if torch.cuda.is_available():
#    DEVICE = torch.device('cuda')
#    print("Using : {}".format(DEVICE))
#else:
#    DEVICE = torch.device('cpu')
#    print("Using : {}".format(DEVICE))  
    
def train_model(model, criterion, optimizer, train_data, train_labels, mini_batch_size):
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

def eval_model(model, criterion, data, labels, roc_curve=False):
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
    if roc_curve:
        curve = roc_curve(labels.detach().cpu(), probs.detach().cpu()[:,1], pos_label=1)
        return eval_loss.item(), acc, AUC, curve
    
    else:return eval_loss.item(), acc, AUC

def get_pred_df(model_dict, data_dict, target_labels_dict):
    """
    Evaluates each model on their targets, then returns a df containing 
    all the stats regarding the predictions.
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
def kfold_cv(model, criterion, optimizer, nb_epochs, lr, kfold,
             mini_batch_size, data, labels, verbose=False):
    """Performs Kfold crossvalidation of a given model."""
    #These will be lists of lists
    kf = KFold(n_splits=kfold, shuffle=True)
    val_result = []
    acc_result = []
    AUC_result = []
    train_result = []

    epoch_print = math.floor(nb_epochs/3)#set an epoch to which we print
    
    for iteration, (train_index, eval_index) in enumerate(kf.split(data)):
        print("Crossvalidating : fold = ", iteration)
        model.reset_parameters()
        train_data, train_labels = data[train_index], labels[train_index]
        eval_data, eval_labels = data[eval_index], labels[eval_index]
        train_temp = []
        val_temp = []
        acc_temp = []
        AUC_temp = []
        #print("Starting training")
        for e in range(nb_epochs):
            train_loss = train_model(model, criterion, optimizer, train_data, train_labels, mini_batch_size)
            val_loss, acc, AUC = eval_model(model, criterion, eval_data, eval_labels)
            train_temp.append(train_loss)
            val_temp.append(val_loss)            
            acc_temp.append(acc)            
            AUC_temp.append(AUC)
            if (e%epoch_print==0|e==nb_epochs)&verbose:
                print("Current stats at epoch = {} :\n"\
                      "Train loss = {}\n Val loss = {}"\
                      "\n Acc = {}\nAUC = {}".format(e,train_loss,
                                                     val_loss,
                                                     acc, AUC))
        train_result.append(train_temp)    
        val_result.append(val_temp)
        acc_result.append(acc_temp)
        AUC_result.append(AUC_temp)
        
    val_result = val_result
    acc_result = acc_result
    AUC_result = AUC_result
    train_result = train_result
    
    return train_result, val_result, acc_result, AUC_result

def batch_run(function, *args):
    for lengths in range(12,17):pass
        
"""  
def train(...):
     if torch.cuda.is_available():
       device = torch.device('cuda')
       print("Using : {}".format(device))
   else:
       device = torch.device('cpu')
       print("Using : {}".format(device))
       #    model = something
    model to device
    
    optimizer = optim.something(model.parameters, lr, ...)
    criterion = nn.CrossEntropyLoss
    etc.
    #Deal with data here
    
    for e in epochs:
        train_model(model, loss, optimizer, ...)
ex : 
def train(..)

    losses = {} a dictionary of lists of losses
    
    for epoch in nb_epochs:
        for length in range(12,17):
            data = feature_dict[length]
            labels = label_dict[length]
            model = model_dict[length]
            loss = train_model(...)
    
""" 
#def train_model(model, criterion, optimizer, train_data, target_labels,
#                epoch, mini_batch_size
#def eval_model(model, criterion, data, labels)

#def batch_train(model_dict, train_data_dict, train_label_dict,
#                criterion, optimizer_dict, lr_dict,
#                nb_epochs, mini_batch_size, kfold=None):
#    """
#    Batch trains all the models
#    """
#    loss_dict = {}
#
#    lengths = range(12,17)
#    for e in nb_epochs:
#        for l in lengths:
#            train_model(model_dict[l], train_data_dict[l], train_label_dict[l],)