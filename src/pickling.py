import pickle
import os

#Small helper file
PATH = os.getcwd()
#Merged dict : [atchley1 ... atchley5, PCA1,...,PCA15]

def save_pkl(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        
def load_pkl(filename):
    with open(filename, 'rb') as f :
        obj = pickle.load(f)
    return obj

def load_all(PATH):
    if 'notebook' in PATH:
        AAidx_Dict = load_pkl('../src/pickles/AAidx_dict.pkl')
        merged_dict = load_pkl('../src/pickles/merged_dict.pkl')
        minmax_aaidx = load_pkl('../src/pickles/minmax_aaidx.pkl')
        minmax_merged = load_pkl('../src/pickles/minmax_merged.pkl')
        minmax_atchley = load_pkl('../src/pickles/minmax_atchley.pkl')
        hla_a = load_pkl('../src/pickles/hla_a.pkl')
        hla_b = load_pkl('../src/pickles/hla_b.pkl')
        
    else :
        AAidx_Dict = load_pkl('./src/pickles/AAidx_dict.pkl')
        merged_dict = load_pkl('./src/pickles/merged_dict.pkl')
        minmax_aaidx = load_pkl('./src/pickles/minmax_aaidx.pkl')
        minmax_merged = load_pkl('./src/pickles/minmax_merged.pkl')
        minmax_atchley = load_pkl('./src/pickles/minmax_atchley.pkl')
        hla_a = load_pkl('./src/pickles/hla_a.pkl')
        hla_b = load_pkl('./src/pickles/hla_b.pkl')
    return AAidx_Dict, merged_dict , minmax_aaidx, minmax_merged, minmax_atchley,  hla_a,  hla_b
        
# Old stuff        
#if 'notebook' in PATH:
#    with open('../src/pickles/AAidx_dict.pkl', 'rb') as f: 
#        AAidx_Dict = pickle.load(f) 
#    with open('../src/pickles/merged_dict.pkl', 'rb') as g: 
#        merged_dict = pickle.load(g)      
#    with open('../src/pickles/minmax_aaidx.pkl','rb') as h:
#        minmax_aaidx = pickle.load(h)
#    with open('../src/pickles/minmax_merged.pkl','rb') as i:
#        minmax_merged = pickle.load(i)
#    with open('../src/pickles/minmax_atchley.pkl','rb') as j:
#        minmax_atchley = pickle.load(j)
#    with open('../src/pickles/hla_a.pkl','rb') as k:
#        hla_a = pickle.load(k)
#    with open('../src/pickles/hla_b.pkl','rb') as l:
#        hla_b = pickle.load(l)        
#else :
#    with open('./src/pickles/AAidx_dict.pkl', 'rb') as f: 
#        AAidx_Dict = pickle.load(f) 
#    with open('./src/pickles/merged_dict.pkl', 'rb') as g: 
#        merged_dict = pickle.load(g) 
#    with open('./src/pickles/minmax_aaidx.pkl','rb') as h:
#        minmax_aaidx = pickle.load(h)
#    with open('./src/pickles/minmax_merged.pkl','rb') as i:
#        minmax_merged = pickle.load(i)
#    with open('./src/pickles/minmax_atchley.pkl','rb') as j:
#        minmax_atchley = pickle.load(j)
#    with open('./src/pickles/hla_a.pkl','rb') as k:
#        hla_a = pickle.load(k)
#    with open('./src/pickles/hla_b.pkl','rb') as l:
#        hla_b = pickle.load(l)   