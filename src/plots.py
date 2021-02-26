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

SAMPLES_ = {12:3182,
            13:6412,
            14:8481,
            15:7040,
            16:2846,}

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
def load_losses(PATH, keys = [12,13,14,15,16]):
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

def get_losses_df(PATH, keys = [12,13,14,15,16]):
    train, val, accs, aucs, f1 = load_losses(PATH,keys)
    seqlist = []
    epochlist = []
    for k in keys:
        seqlist += [k]*len(train[k])
        epochlist += range(len(train[k]))

    result = pd.DataFrame(data =[] ,columns=['epoch','seqlen','value','type'])
    names = ['train_loss','val_loss','val_accs','val_aucs','val_f1']
    for index, tmp in enumerate([train, val, accs, aucs, f1]):
        tmp_df = pd.DataFrame.from_dict(tmp, orient = 'columns')
        tmp_df['epoch'] = range(len(tmp_df))
        melt = tmp_df.melt(id_vars='epoch', var_name = 'seqlen', value_name = 'value')
        melt['type'] = names[index]
        #result = pd.merge(result, melt, left_index=True, right_index=True, suffixes=('', '_delme'))
        #result = result[[c for c in result.columns if not c.endswith('_delme')]]
        result = pd.concat([result,melt])
    
    return result
        

def plot_loss(train_dict, val_dict, kcv=False, save = 'losses.jpg', folder = None):
    keys = train_dict.keys()
    num = len(keys)
    fig, axes = plt.subplots(EPOCHLAYOUT[num][0],EPOCHLAYOUT[num][1], figsize = EPOCHLAYOUT[num][2])
    if num > 1: 
        ax = axes.ravel() 
        for index, ll in enumerate(keys):
            if kcv == True : 
                train_mean = np.mean(train_dict[ll],axis=0)
                train_err = np.var(train_dict[ll], axis=0)
                val_mean = np.mean(val_dict[ll],axis=0)
                val_err = np.var(val_dict[ll], axis=0)
                x = range(len(train_mean))      
                ax[index].errorbar(x, train_mean, yerr=train_err,fmt= 'b-', lw = 0.5, label = 'Train', capsize=.5, capthick = .5)
                ax[index].errorbar(x, val_mean, yerr=val_err, fmt='r-', lw = 0.5, label = 'Validation', capsize=.5, capthick = .5)
            else:    
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
        return fig, axes
    if folder is not None:
        plt.savefig(os.path.join(folder,save))
        return fig, axes
    else:
        plt.savefig(OUTPATH+save)
        return fig, axes


def plot_roc_curve(curve_dict, save = 'roc_curves.jpg', folder=None):
    keys = curve_dict.keys()
    num = len(keys)
    fig, axes = plt.subplots(SQRLAYOUT[num][0],SQRLAYOUT[num][1], figsize = SQRLAYOUT[num][2])
    if num > 1: 
        ax = axes.ravel() 
        for index, ll in enumerate(keys):
            pts = np.linspace(0,1, 100)
            ax[index].plot(curve_dict[ll][1][0], curve_dict[ll][1][1],
                           'm-', lw = 0.8, label = 'AUC={:.4f}'.format(curve_dict[ll][0]))
            ax[index].plot(pts, pts, 'k--', lw = '0.4', label = 'random')
            ax[index].legend(loc='best')
            ax[index].set_title('ROC_AUC on test set for L = {}'\
                                ', total {} samples (pos+neg)'.format(ll,SAMPLES_[ll]))
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
        axes.set_title('ROC_AUC on test set for L = {}'\
                       ', total {} samples (pos+neg)'.format(ll,SAMPLES_[ll]))
        axes.set_xlabel('FPR')
        axes.set_ylabel('TPR')
    
    fig.suptitle('ROC AUC curve for each model on test set:', fontsize = 18, fontweight='bold')
    fig.subplots_adjust(top=.95)
    fig.tight_layout(rect=[0,0,1,.99])
    if save == None:
        return fig, axes
    if folder is not None:
        plt.savefig(os.path.join(folder,save))
        return fig, axes

    else:
        plt.savefig(OUTPATH+save)
        return fig, axes
        
        
def plot_accs(accuracy_dict, AUC_dict, F1_dict, 
              kcv=False, save = 'accs.jpg', folder = None):
    keys = AUC_dict.keys()
    num = len(keys)
    fig, axes = plt.subplots(EPOCHLAYOUT[num][0],EPOCHLAYOUT[num][1], figsize = EPOCHLAYOUT[num][2])
    if num > 1: 
        ax = axes.ravel() 
        for index, ll in enumerate(keys):
            if kcv == True : 
                acc_mean = np.mean(accuracy_dict[ll],axis=0)
                acc_err = np.var(accuracy_dict[ll], axis=0)
                
                AUC_mean = np.mean(AUC_dict[ll],axis=0)
                AUC_err = np.var(AUC_dict[ll], axis=0)
                
                F1_mean = np.mean(F1_dict[ll],axis=0)
                F1_err = np.var(F1_dict[ll], axis=0)

                x = range(len(acc_mean))      
                ax[index].errorbar(x, acc_mean, yerr=acc_err,fmt= 'c-', lw = 0.5, label = 'Accuracy', capsize=.5, capthick = .5)
                ax[index].errorbar(x, AUC_mean, yerr=AUC_err, fmt='g-', lw = 0.5, label = 'AUC', capsize=.5, capthick = .5)
                ax[index].errorbar(x, F1_mean, yerr=F1_err, fmt='y-', lw = 0.5, label = 'F1-score', capsize=.5, capthick = .5)
            else:   
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
            
    fig.suptitle('Prediction stats for each model during training:', fontsize = 18, fontweight='bold')
    fig.subplots_adjust(top=.95)
    fig.tight_layout(rect=[0,0,1,.99])

    if save == None:
        return            
    if folder is not None:
        plt.savefig(os.path.join(folder,save))
    else:
        plt.savefig(OUTPATH+save)   
        
def plot_PPV(df, save='PPV.jpg', folder =None):
    """Need to use get_preds_df"""
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
            ax[i].set_title('PPV vs (TP+FP) for L ={}'\
                            ', total {} samples (pos+neg)'.format(sl, SAMPLES_[sl]), weight='bold')
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
        axes.set_title('PPV vs (TP+FP) for L ={}'\
                            ', total {} samples (pos+neg)'.format(sl, SAMPLES_[sl]), weight='bold')
        axes.set_ylabel('PPV')
        axes.set_xlabel('Number of predictions (logscale)')
        
    fig.suptitle('PPV for each model on test set:', fontsize = 18, fontweight='bold')
    fig.subplots_adjust(top=.95)
    fig.tight_layout(rect=[0,0,1,.99])

    if save == None:
        return            
    if folder is not None:
        plt.savefig(os.path.join(folder,save))
    else:
        plt.savefig(OUTPATH+save)
        
#def plot_loss_seaborn(df)


def plot_roc_comparison(curves_dict1, curves_dict2, name, save='roc_comparison.jpg', folder =None):
    fig, axes = plt.subplots(2, 3, figsize=(30,20))
    ax = axes.ravel() 
    for index, ll in enumerate(range(12,17)):
        pts = np.linspace(0,1, 100)
        ax[index].plot(curves_dict1[ll][1][0], curves_dict1[ll][1][1],
                       'm-', lw = 0.8, label = 'DeepCAT : AUC={:.4f}'.format(curves_dict1[ll][0]))
        ax[index].plot(curves_dict2[ll][1][0], curves_dict2[ll][1][1],
                       'c-', lw = 0.8, label = 'Richie : AUC={:.4f}'.format(curves_dict2[ll][0]))
        ax[index].plot(pts, pts, 'k--', lw = '0.4', label = 'random')
        ax[index].legend(loc='best')
        ax[index].set_title('ROC_AUC on new set {} L = {}'.format(name,ll))
        ax[index].set_xlabel('FPR')
        ax[index].set_ylabel('TPR')
    fig.suptitle('Comparison of ROC AUC curves on {}'.format(name), fontsize = 18, fontweight='bold')
    fig.subplots_adjust(top=.95)
    fig.tight_layout(rect=[0,0,1,.99])
    fig.delaxes(ax[-1])
        
    if save == None:
        return            
    if folder is not None:
        plt.savefig(os.path.join(folder,save), dpi=180)
    else:
        plt.savefig(os.path.join(os.getcwd(),save), dpi=180)
        
from statsmodels.stats.weightstats import ttest_ind

def boxplot_cs(total_df, pv=False):
    uniques=total_df['data'].unique()
    n = len(uniques)
    fig, axes = plt.subplots(n,1, figsize=(15,8*n))
    
    for index, a in enumerate(axes.ravel()):
        #Getting dataset subgraph
        x = uniques[index]
        data = total_df.query('data==@x')
        #Getting colorpalette
        n_hues = len(data.type.unique())
        sns.set_palette('coolwarm', n_colors =n_hues)

        sns.boxplot(data=data, x='model', hue='type', y = 'cancer_score', ax = a)
        sns.swarmplot(data=data, x='model', hue='type', y = 'cancer_score', ax = a, edgecolor='black', linewidth=0.5)
        a.set_title('Comparison of predicted patient cancer scores for data = {} \nusing DeepCAT re-trained model'\
                    ' vs PyTorch (Richie) implementation'.format(x))
        if pv ==True:
            pvals = []
            top = data.cancer_score.max()
            a.set_ylim(data.cancer_score.min()-0.01, top+0.02)
            print('Here PVAL')
            y1, y2, y3 = top+0.005, top+0.01, top+0.015
            #Getting models and plotting
            n_models = data.model.unique()
            for i, mod in enumerate(n_models):
                pval = ttest_ind(data.query('model==@mod&type=="control"')['cancer_score'],
                                 data.query('model==@mod&type=="cancer"')['cancer_score'])
                a.plot([i-0.2, i-0.2, i+0.2, i+0.2], [y1, y2, y2, y1], lw=1.5, c='k')
                a.text(i, y3, "p = {pv:.3e}".format(pv=pval[1]),
                       ha='center', va='center', color='k')