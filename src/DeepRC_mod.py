# -*- coding: utf-8 -*-
"""
Network architectures

See `deeprc/examples/` for examples.

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import numpy as np
import torch
import torch.nn as nn
import torch.jit as jit
from typing import List

#Using CNN as Sequence Embedder
class SequenceEmbeddingCNN(nn.Module):
    def __init__(self, n_input_features: int, kernel_size: int = 4, n_kernels: int = 32, n_layers: int = 1, dropout=False):
        """Sequence embedding using 1D-CNN (`h()` in paper)
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        n_input_features : int
            Number of input features per sequence position
        kernel_size : int
            Size of 1D-CNN kernels
        n_kernels : int
            Number of 1D-CNN kernels in each layer
        n_layers : int
            Number of 1D-CNN layers
        """
        super(SequenceEmbeddingCNN, self).__init__()
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.n_layers = n_layers
        if dropout==True:
            self.dropout = nn.Dropout(0.3)
        else : self.dropout = nn.Identity()
        if self.n_layers <= 0:
            raise ValueError(f"Number of layers n_layers must be > 0 but is {self.n_layers}")
        
        # CNN layers
        network = []
        for i in range(self.n_layers):
            conv = nn.Conv1d(in_channels=n_input_features, out_channels=self.n_kernels, kernel_size=self.kernel_size,
                             bias=True)
            conv.weight.data.normal_(0.0, np.sqrt(1 / np.prod(conv.weight.shape)))
            network.append(conv)
            network.append(nn.SELU(inplace=True))
            #HERE ADDED NEW
            network.append(nn.Dropout(p=0.3))
            n_input_features = self.n_kernels
        
        self.network = torch.nn.Sequential(*network)
    
    def forward(self, inputs, *args, **kwargs):
        """Apply sequence embedding CNN to inputs in NLC format.
        
        Parameters
        ----------
        inputs: torch.Tensora
            Torch tensor of shape (n_sequences, n_sequence_positions, n_input_features).
        
        Returns
        ---------
        max_conv_acts: torch.Tensor
            Sequences embedded to tensor of shape (n_sequences, n_kernels)
        """
        inputs = torch.transpose(inputs, 1, 2)  # NLC -> NCL
        # Apply CNN
        conv_acts = self.network(inputs)
        # Take maximum over sequence positions (-> 1 output per kernel per sequence)
        max_conv_acts, _ = conv_acts.max(dim=-1)
        return max_conv_acts
    
class AttentionNetwork(nn.Module):
    def __init__(self, n_input_features: int, n_layers: int = 2, n_units: int = 32, dropout=False):
        """Attention network (`f()` in paper) as fully connected network.
         Currently only implemented for 1 attention head and query.
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        n_input_features : int
            Number of input features
        n_layers : int
            Number of attention layers to compute keys
        n_units : int
            Number of units in each attention layer
        """
        super(AttentionNetwork, self).__init__()
        self.n_attention_layers = n_layers
        self.n_units = n_units
        if dropout == True:
            self.dropout = nn.Dropout(0.3)
        else : self.dropout = nn.Identity()
            
        fc_attention = []
        for _ in range(self.n_attention_layers):
            att_linear = nn.Linear(n_input_features, self.n_units)
            att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
            fc_attention.append(att_linear)
            fc_attention.append(nn.SELU())
            #HERE ADDED NEW
            fc_attention.append(self.dropout)
            n_input_features = self.n_units
        
        att_linear = nn.Linear(n_input_features, 1)
        att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
        fc_attention.append(att_linear)
        self.attention_nn = torch.nn.Sequential(*fc_attention)
    
    def forward(self, inputs):
        """Apply single-head attention network.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Torch tensor of shape (n_sequences, n_input_features)
        
        Returns
        ---------
        attention_weights: torch.Tensor
            Attention weights for sequences as tensor of shape (n_sequences, 1)
        """
        attention_weights = self.attention_nn(inputs)
        return attention_weights

class OutputNetwork(nn.Module):
    def __init__(self, n_input_features: int, n_output_features: int = 1, n_layers: int = 1, n_units: int = 32, dropout = False):
        """Output network (`o()` in paper) as fully connected network
        
        See `deeprc/examples/` for examples.
        
        Parameters
        ----------
        n_input_features : int
            Number of input features
        n_output_features : int
            Number of output features
        n_layers : int
            Number of layers in output network (in addition to final output layer)
        n_units : int
            Number of units in each attention layer
        """
        super(OutputNetwork, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        
        output_network = []
        for _ in range(self.n_layers-1):
            o_linear = nn.Linear(n_input_features, self.n_units)
            o_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(o_linear.weight.shape)))
            output_network.append(o_linear)
            output_network.append(nn.SELU())
            n_input_features = self.n_units
        
        o_linear = nn.Linear(n_input_features, n_output_features)
        o_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(o_linear.weight.shape)))
        output_network.append(o_linear)
        self.output_nn = torch.nn.Sequential(*output_network)
        self.sig = nn.Sigmoid()
        #HERE ADDED NEW
        if dropout == True:
            self.dropout = nn.Dropout(p=0.3)
        else:
            self.dropout = nn.Identity()
        
    def forward(self, inputs):
        """Apply output network to `inputs`.
        
        Parameters
        ----------
        inputs: torch.Tensor
            Torch tensor of shape (n_samples, n_input_features).
        
        Returns
        ---------
        prediction: torch.Tensor
            Prediction as tensor of shape (n_samples, n_output_features).
        """
        #HERE ADDED NEW
        predictions = self.dropout(self.sig(self.output_nn(inputs)))
        return predictions

class DeepRC_mod(nn.Module): 
    
    def __init__(self, #20 for the initial onehot encode dim
                 embedding_net: torch.nn.Module = SequenceEmbeddingCNN(n_input_features = 20,
                                                                       kernel_size = 4, n_kernels=32,
                                                                       n_layers = 2),
                 attention_net: torch.nn.Module = AttentionNetwork(n_input_features = 32, 
                                                                   n_layers=2, n_units = 32),
                 output_net_A: torch.nn.Module = OutputNetwork(n_input_features=32, 
                                                             n_output_features = 21,
                                                              n_layers = 2, n_units = 32),
                 output_net_B: torch.nn.Module = OutputNetwork(n_input_features=32, 
                                                             n_output_features = 40,
                                                              n_layers = 2, n_units = 32)
                ):
        super(DeepRC_mod, self).__init__()
        self.embedding = embedding_net.to(dtype=torch.float32)
        self.attention = attention_net.to(dtype=torch.float32)
        self.out_A = output_net_A.to(dtype=torch.float32)
        self.out_B = output_net_B.to(dtype=torch.float32)
        
    def forward(self, x, n_per_bag):
        seq_embed = self.embedding(x)
        seq_attention = self.attention(seq_embed)
        # BUT softmax(attn_weight) must be done PER BAG. Given we have the number of sequences,
        #we treat the input (A sequence of bags) sequentially : 
        
        x = []
        start_i = 0
        # n_per_bag stores the number of sequences per bag, so we can use it to slice the attention and embedding
        for n_seqs in n_per_bag : 
            #SLICE AND SOFTMAX OVER THE SLICE
            attention_slice = torch.softmax(seq_attention[start_i:start_i+n_seqs], dim=0)
            embedding_slice = seq_embed[start_i:start_i+n_seqs]
            
            embedding_attention = embedding_slice * attention_slice
            #Weighted sum over the features. The Weight is from the attention
            x.append(embedding_attention.sum(dim=0))
            start_i += n_seqs
            del embedding_attention
            
        x = torch.stack(x, dim = 0)
        pred_a = self.out_A(x)
        pred_b = self.out_B(x)
        return pred_a, pred_b
