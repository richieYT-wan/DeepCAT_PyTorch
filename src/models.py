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
        self.pool1 = nn.MaxPool2d(kernel_size = (1,2), stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (1,2))
        self.pool2 = nn.MaxPool2d(kernel_size = (1,2), stride=1)
        #Getting the dimension after convolutions
        self.length = seq_len #Storing this value, in case SUBJECT TO CHANGE
        self.dummy_param = nn.Parameter(torch.empty(0))
        if seq_len<15:
            self.nb_units = 16*1*2 
        elif seq_len >=15:
            self.nb_units = 16*1*3 
        #Linear/Dense layers
        self.fc1 = nn.Linear(self.nb_units, 10)
        self.fc2 = nn.Linear(10,2)
        self.dropout= nn.Dropout(0.2)

    def forward(self, x):
        #Conv -> ReLU -> MaxPool
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #Linear->ReLU->Dropout
        x = x.view(-1, self.nb_units) #reshaping after convolution
        x = self.dropout(F.relu(self.fc1(x))) 
        x = self.dropout(F.relu(self.fc2(x))) #Getting binary logits

        label = x.argmax(1)
        probs = x.softmax(1)

        return probs, label
        