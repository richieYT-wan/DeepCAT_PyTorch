from src.models import *
from src.preprocessing import *
from src.train_eval_helpers import *
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']= 180
sns.set_style('darkgrid')
#sns.set_palette("coolwarm", n_colors=2)

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
    5: [2, 3, (24,24)]
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
                            
def plot_loss(train_dict, val_dict, keys, save = 'losses.jpg'):
    num = len(keys)
    fig, axes = plt.subplots(EPOCHLAYOUT[num][0],EPOCHLAYOUT[num][1], figsize = EPOCHLAYOUT[num][2])
    if num > 1: 
        ax = axes.ravel() 
        for index, ll in enumerate(keys):
            ax[index].plot(train_dict[ll], 'b-', lw = 0.5, label = 'Train')
            ax[index].plot(val_dict[ll], 'r-', lw = 0.5, label = 'Validation')
            ax[index].legend(loc='best')
            ax[index].set_title('Train/Val Losses for during training L = {}'.format(ll))
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
        axes.set_title('Train/Val Losses for during training L = {}'.format(ll),fontweight='bold')
        axes.set_xlabel('epoch')
        axes.set_ylabel('Loss')
    plt.savefig(save)

def plot_roc_curve(curve_dict, keys, save = 'roc_curves.jpg'):
    num = len(keys)
    fig, axes = plt.subplots(SQRLAYOUT[num][0],SQRLAYOUT[num][1], figsize = SQRLAYOUT[num][2])
    if num > 1: 
        ax = axes.ravel() 
        for index, ll in enumerate(keys):
            pts = np.linspace(0,1, (math.floor(len(curve_dict[ll][1][0])/100)))
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
        pts = np.linspace(0,1, (math.floor(len(curve_dict[ll][1][0])/100)))
        axes.plot(curve_dict[ll][1][0], curve_dict[ll][1][1],
                'm-', lw = 0.8, label = 'AUC={}'.format(curve_dict[ll][0]))
        axes.plot(pts, pts, 'k--', lw = '0.4', label = 'random')
        axes.legend(loc='best')
        axes.set_title('ROC_AUC on test set for L = {}'.format(ll))
        axes.set_xlabel('FPR')
        axes.set_ylabel('TPR')
        
    plt.savefig(save)