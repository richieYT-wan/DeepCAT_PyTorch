from src.models import *
from src.preprocessing import *
from src.train_eval_helpers import *
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
mpl.rcParams['figure.dpi']= 180
sns.set_style('darkgrid')
#sns.set_palette("coolwarm", n_colors=2)

os.path.abspath(os.path.join(os.path.os.getcwd(), os.pardir))


OUTPATH = os.path.join(os.path.abspath(os.path.join(os.path.os.getcwd(), os.pardir)),'figures/')

EPOCHLAYOUT = {
    1: [1, 1, (8,6)],
    2: [1, 2, (16,6)],
    3: [1, 3, (24,6)],
    4: [2, 2, (16,12)],
    5: [2, 3, (24,12)]
}

SQRLAYOUT = {
    1: [1, 1, (8,8)],
    2: [1, 2, (16,8)],
    3: [1, 3, (24,8)],
    4: [2, 2, (16,16)],
    5: [2, 3, (24,16)]
}    
def load_losses(keys, PATH):
    
    
    fns=['train_losses_dict.pkl','val_losses_dict.pkl', 
         'val_accs_dict.pkl','val_aucs_dict.pkl','val_f1_dict.pkl']
    z = []
    for index, name in enumerate(fns):
        picklename = os.path.join(PATH, name)
        with open(picklename, 'rb') as f:
            temp = pickle.load(f)
            z.append(temp)

    #Returning only the subset corresponding to the input keys
    train_loss_dict = dict((k, z[0][k]) for k in keys) 
    val_loss_dict = dict((k, z[1][k]) for k in keys)
    val_accs_dict = dict((k, z[2][k]) for k in keys)
    val_aucs_dict = dict((k, z[3][k]) for k in keys)
    val_f1_dict = dict((k, z[4][k]) for k in keys) 
    print("Values loaded")
    return train_loss_dict, val_loss_dict, val_accs_dict, val_aucs_dict, val_f1_dict 
                            
def plot_loss(train_dict, val_dict, keys, save = 'losses.jpg', folder = None):
    num = len(keys)
    fig, axes = plt.subplots(EPOCHLAYOUT[num][0],EPOCHLAYOUT[num][1], figsize = EPOCHLAYOUT[num][2])
    if num > 1: 
        ax = axes.ravel() 
        for index, ll in enumerate(keys):
            ax[index].plot(train_dict[ll], 'b-', lw = 0.5, label = 'Train')
            ax[index].plot(val_dict[ll], 'r-', lw = 0.5, label = 'Validation')
            ax[index].legend(loc='best')
            ax[index].set_title('Train/Val Losses during training for L = {}'.format(ll))
            ax[index].set_xlabel('epoch')
            ax[index].set_ylabel('Loss')
        if num == 5 : fig.delaxes(ax[-1])
    if num == 1:
        ll=keys[0]
        print(ll)
        print(train_dict.keys())
        axes.plot(train_dict[ll], 'b-', lw = 0.5, label = 'Train')
        axes.plot(val_dict[ll], 'r-', lw = 0.5, label = 'Validation')
        axes.legend(loc='best')
        axes.set_title('Train/Val Losses during training for L = {}'.format(ll),fontweight='bold')
        axes.set_xlabel('epoch')
        axes.set_ylabel('Loss')
    if save == None:
        return
    if folder is not None:
        plt.savefig(os.path.join(folder,save))
    else:
        plt.savefig(OUTPATH+save)


def plot_roc_curve(curve_dict, keys, save = 'roc_curves.jpg', folder=None):
    num = len(keys)
    fig, axes = plt.subplots(SQRLAYOUT[num][0],SQRLAYOUT[num][1], figsize = SQRLAYOUT[num][2])
    if num > 1: 
        ax = axes.ravel() 
        for index, ll in enumerate(keys):
            pts = np.linspace(0,1, 100)
            ax[index].plot(curve_dict[ll][1][0], curve_dict[ll][1][1],
                           'm-', lw = 0.8, label = 'AUC={}'.format(curve_dict[ll][0]))
            ax[index].plot(pts, pts, 'k--', lw = '0.4', label = 'random')
            ax[index].legend(loc='best')
            ax[index].set_title('ROC_AUC on test set for L = {}'.format(ll))
            ax[index].set_xlabel('FPR')
            ax[index].set_ylabel('TPR')
        if num == 5 : fig.delaxes(ax[-1])
            
    if num == 1:
        ll=keys[0]
        pts = np.linspace(0,1, 100)
        axes.plot(curve_dict[ll][1][0], curve_dict[ll][1][1],
                'm-', lw = 0.8, label = 'AUC={}'.format(curve_dict[ll][0]))
        axes.plot(pts, pts, 'k--', lw = '0.4', label = 'random')
        axes.legend(loc='best')
        axes.set_title('ROC_AUC on test set for L = {}'.format(ll))
        axes.set_xlabel('FPR')
        axes.set_ylabel('TPR')
    if save == None:
        return
    if folder is not None:
        plt.savefig(os.path.join(folder,save))
    else:
        plt.savefig(OUTPATH+save)
def plot_accs(accuracy_dict, AUC_dict, F1_dict, keys, 
              save = 'accs.jpg', folder = None):
    num = len(keys)
    fig, axes = plt.subplots(EPOCHLAYOUT[num][0],EPOCHLAYOUT[num][1], figsize = EPOCHLAYOUT[num][2])
    if num > 1: 
        ax = axes.ravel() 
        for index, ll in enumerate(keys):
            ax[index].plot(accuracy_dict[ll], 'c-', lw = 0.5, label = 'Accuracy')
            ax[index].plot(AUC_dict[ll], 'g-', lw = 0.5, label = 'AUC')
            ax[index].plot(F1_dict[ll], 'y-', lw = 0.5, label = 'F1-score')
            
            ax[index].legend(loc='best')
            ax[index].set_title('Prediction stats during training for L = {}'.format(ll))
            ax[index].set_xlabel('epoch')
            ax[index].set_ylabel('%')
        if num == 5 : fig.delaxes(ax[-1])
    if num == 1:
        ll=keys[0]
        #print(ll)
        #print(train_dict.keys())
        axes.plot(accuracy_dict[ll], 'c-', lw = 0.5, label = 'Accuracy')
        axes.plot(AUC_dict[ll], 'g-', lw = 0.5, label = 'AUC')
        axes.plot(F1_dict[ll], 'y-', lw = 0.5, label = 'F1-score')
        axes.legend(loc='best')
        axes.set_title('Prediction stats during training for L = {}'.format(ll),fontweight='bold')
        axes.set_xlabel('epoch')
        axes.set_ylabel('%')
    if save == None:
        return            
    if folder is not None:
        plt.savefig(os.path.join(folder,save))
    else:
        plt.savefig(OUTPATH+save)   
        
def plot_PPV(df, save='PPV.jpg', folder =None):
    keys = df['seqlen'].unique()
    num = len(keys) #Number of keys (plots needed)
    fig, axes = plt.subplots(EPOCHLAYOUT[num][0],EPOCHLAYOUT[num][1], figsize = EPOCHLAYOUT[num][2])
    
    xs=df.sort_values('prob_cancer',ascending=False)\
         .groupby('seqlen')\
         .apply(lambda x: np.cumsum(x['tp'].values)/np.cumsum(np.ones(len(x['tp']))))

    perfect = df.sort_values('y_true',ascending=False)\
                .groupby('seqlen')\
                .apply(lambda x: np.cumsum(x['y_true'].values)/np.cumsum(np.ones(len(x['y_true']))))
    
    line = ['c--','r--','g--', 'b--', 'm--']
    if num > 1 :
        ax = axes.ravel()
        for i, sl in enumerate(xs.index):
            ax[i].semilogx(range(0,len(df.query('seqlen==@sl')['tp'])), xs[sl], 
                           line[i], lw=1, label='Prediction : L={}'.format(sl))
            ax[i].semilogx(range(0,len(df.query('seqlen==@sl')['y_true'])), perfect[sl],
                           'k-.', lw=0.5, label = '"Perfect prediction"')
            ax[i].legend(loc='best')
            ax[i].set_title('PPV vs (TP+FP) for L ={}'.format(sl), weight='bold')
            ax[i].set_ylabel('PPV')
            ax[i].set_xlabel('Number of predictions (logscale)')
        #a.ygrid()
            ax[i].grid(True, which="both", ls="-.", color='0.8')
        if num == 5 : fig.delaxes(ax[-1])
    
    if num == 1:
        axes.semilogx(range(0,len(df.query('seqlen==@sl')['tp'])), xs[sl], 
                      line[i], lw=1, label='Prediction : L={}'.format(sl))
        axes.semilogx(range(0,len(df.query('seqlen==@sl')['y_true'])), perfect[sl],
                      'k-.', lw=0.5, label = '"Perfect prediction"')
        axes.legend(loc='best')
        axes.set_title('PPV vs (TP+FP) for L ={}'.format(sl), weight='bold')
        axes.set_ylabel('PPV')
        axes.set_xlabel('Number of predictions (logscale)')
        
    if save == None:
        return            
    if folder is not None:
        plt.savefig(os.path.join(folder,save))
    else:
        plt.savefig(OUTPATH+save)