"""File for protein language model (PLM) models"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer
"""Separated into 2 modules, in case I want to use HLA-embedding + CNN + linear or simply HLA-pooled + linear"""

class bert_transformer(nn.Module):  
    def __init__(self, freeze=True):
        super(bert_transformer, self).__init__()
        #Just the number of "classes" (allele) present in the dataset for HLA
        #Weight sharing for transformer
        
        self.transformer = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.dim = self.transformer.config.hidden_size
        if freeze==True:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        
        output = self.transformer(input_ids = input_ids,
                                  attention_mask = attention_mask)[1] #[1] = the pooled output (shape (N, 1024)). [0] for embedding (shape (N, max_len, 1024))
        return output
    
class final_prediction(nn.Module):
    def __init__(self, in_dim=1024, p_drop=0.4):
        super(final_prediction,self).__init__()
        #Hardcoded class numbers for now
        self.n_classes_A = 21
        self.n_classes_B = 40 
        self.dropout = nn.Dropout(p_drop)
        #For now, simple linear 1024 --> n_class one layer model
        self.linear_A = nn.Linear(in_dim, self.n_classes_A)
        self.linear_B = nn.Linear(in_dim, self.n_classes_B)
        #self.bn_A = nn.BatchNorm1d(21)
        #self.bn_b = nn.BatchNorm1d(40)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        out_a = self.dropout(self.sig(self.linear_A(x)))
        out_b = self.dropout(self.sig(self.linear_B(x)))
        return out_a, out_b
    
class HLAPredictor(nn.Module):
    """
    Implementation subject to change depending on whether I use pooled output or embeddings for transformer.
    For now, assumes that I use the pooled output."""
    def __init__(self, freeze = True, p_drop_final=0.4):
        super(HLAPredictor, self).__init__()
        self.transformer = bert_transformer(freeze=freeze)
        self.predictor = final_prediction(in_dim=self.transformer.dim, 
                                          p_drop = p_drop_final)
        
    def forward(self, input_ids, attention_mask):
        transformed = self.transformer(input_ids, attention_mask)
        predicted_a, predicted_b = self.predictor(transformed)
        return predicted_a, predicted_b
