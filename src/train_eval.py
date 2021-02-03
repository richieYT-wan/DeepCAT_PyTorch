from src.preprocessing import *
from src.models import deepcat_cnn
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.   data import BatchSampler, RandomSampler    
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

#if torch.cuda.is_available():
#    DEVICE = torch.device('cuda')
#    print("Using : {}".format(DEVICE))
#else:
#    DEVICE = torch.device('cpu')
#    print("Using : {}".format(DEVICE))
    
def train_model(model, criterion, optimizer, train_data, train_labels,
                epoch, mini_batch_size):
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
    #train_loss /= train_data.size(0)
    #if e%200==0:
     #   print("Loss at epoch {e} : {l}".format(e=epoch,l=train_loss))
    return train_loss/train_data.size(0)

def eval_model(model, criterion, data, labels, roc=False):
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
    accs = accuracy_score(labels.cpu(), predictions.cpu())
    AUCs = roc_auc_score(one_hot, probs.detach().cpu())
    #if roc==True:
    #    curve = metrics.roc_curve(labels.cpu(), probs.detach().cpu())
    #    return (eval_loss.item()/labels.size(0)), accs, AUCs, curve
    return (eval_loss.item()/labels.size(0)), accs, AUCs

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