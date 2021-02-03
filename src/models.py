import torch
import torch.nn as nn
import torch.nn.functional as F 

class deepcat_cnn(torch.nn.Module):
    """
    PyTorch version of DeepCAT CNN
    seq_len represents the length (# of AAs) of the CDR3 region (L = 12, 13, ..., 16)
    When initializing CNN_CDR3, the L should be specified 
    """ 
    def __init__(self, seq_len):
        super(deepcat_cnn, self).__init__()
        #Convolutions
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(15,2))
        self.pool1 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (1,2))
        self.pool2 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        #Getting the dimension after convolutions
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.length = seq_len
        
        #if seq_len<15: only correct if no stride on maxpool
        #    self.nb_units = 16*1*2 
        #elif seq_len >=15:
        #    self.nb_units = 16*1*3 
        
        #Linear/Dense layers
        self.fc1 = nn.Linear(16*(self.length-4), 10)
        self.fc2 = nn.Linear(10,2)
        self.dropout= nn.Dropout(0.4)

    def forward(self, x):
        #Conv -> ReLU -> MaxPool
        #print("input",x.shape)
        x = F.relu(self.conv1(x))
        #print("after conv1", x.shape)
        x = self.pool1(x)
        #print("after pool1",x.shape)
        x = F.relu(self.conv2(x))
        #print("After conv2", x.shape)
        x = self.pool2(x)
        #print("After pool2", x.shape)
        #Linear->ReLU->Dropout
        #print("Before reshape",x.shape)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) #reshaping after convolution
        #print("reshaped",x.shape)
        x = self.dropout(F.relu(self.fc1(x))) 
        x = self.dropout(F.relu(self.fc2(x))) #Getting binary logits
        
        #label = x.argmax(1)
        #probs = x.softmax(1)

        return x #probs, label