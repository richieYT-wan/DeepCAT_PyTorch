import torch
import torch.nn as nn
import torch.nn.functional as F 
import os 

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
        self.name = 'deepcat_cnn_'+str(seq_len)
        #Linear/Dense layers
        self.fc1 = nn.Linear(16*(self.length-4), 10)
        self.fc2 = nn.Linear(10,2)
        self.dropout= nn.Dropout(0.4)
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                layer.zero_grad()
                
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
        #THERES A RELU HERE THAT SHOULD BE ::++++ TODO
        x = self.dropout(F.relu(self.fc2(x))) #Getting binary logits
        
        predictions = x.argmax(1)
        probabilities = x.softmax(1)

        return x, predictions, probabilities

def get_models(keys):
    model_dict = {}
    for key in keys:
        model = deepcat_cnn(key)
        model_dict[key]=model
        del model
    return model_dict

def load_models(keys, PATH):
    model_dict = get_models(keys)
    files = os.listdir(PATH)
    files = [f for f in files if '.pth.tar' in f]
    
    for key in keys:
        pattern = 'deepcat_cnn_'+str(key)
        chkpt = {}
        for fn in files:
            if pattern in fn:
                name = os.path.join(PATH,fn)
                chkpt = torch.load(name)
            else:continue
        model_dict[key].load_state_dict(chkpt['state_dict'])
    print("Models & weights loaded")
    return model_dict

def model_info(keys, PATH):
    model_dict = get_models(keys)
    files = os.listdir(PATH)
    files = [f for f in files if '.pth.tar' in f]
    
    for key in keys:
        pattern = 'deepcat_cnn_'+str(key)
        chkpt = {}
        for fn in files:
            if pattern in fn:
                name = os.path.join(PATH,fn)
                chkpt = torch.load(name)
            else:continue
        print('L = {}'.format(key))
        for (k,v) in chkpt.items():
            if k == 'state_dict':continue
            else:print("\t{} : {}".format(k,v))
            