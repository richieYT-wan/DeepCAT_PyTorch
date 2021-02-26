import torch
import torch.nn as nn
import torch.nn.functional as F 
import os 

################ ARCHITECTURES #####################
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
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #Linear->ReLU->Dropou
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) #reshaping after convolution
        x = self.dropout(F.relu(self.fc1(x))) 
        #THERES A RELU HERE THAT SHOULD BE ::++++ TODO
        x = self.dropout(F.relu(self.fc2(x))) #Getting binary logits
        
        predictions = x.argmax(1)
        probabilities = x.softmax(1)

        return x, predictions, probabilities

class richie_net(torch.nn.Module):
    """
    PyTorch version of DeepCAT CNN
    seq_len represents the length (# of AAs) of the CDR3 region (L = 12, 13, ..., 16)
    When initializing CNN_CDR3, the L should be specified 
    """ 
    def __init__(self, seq_len):
        super(richie_net, self).__init__()
        #Convolutions
        self.conv1 = nn.Conv2d(1,8, kernel_size = (10,3))
        self.pool1 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,1))
        self.conv2 = nn.Conv2d(8,16, kernel_size = (10,3))
        self.pool2 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,1))
        #Getting the dimension after convolutions
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.length = seq_len
        self.name = 'richie_net_'+str(seq_len)
        #Linear/Dense layers
        self.fc1 = nn.Linear(16*(self.length-6)*2, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,10)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10,2)
        self.dropout= nn.Dropout(0.4)
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                layer.zero_grad()
                
    def forward(self, x):
        #Conv -> ReLU -> MaxPool
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #Linear->ReLU->Dropout
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) #reshaping after convolution
        #FC layers
        x = self.dropout(self.bn1(F.relu(self.fc1(x)))) 
        x = self.dropout(self.bn2(F.relu(self.fc2(x)))) #Getting binary logits
        x = self.dropout(F.relu(self.fc3(x)))
        predictions = x.argmax(1)
        probabilities = x.softmax(1)

        return x, predictions, probabilities
    
class variable_net(torch.nn.Module):
    """
    Variable filter lengths, based on the fact that the sequences have well conserved start and end regions of lengths 4 aa. each
    
    """ 
    def __init__(self, seq_len):
        super(variable_net, self).__init__()
        #Convolutions
        self.length = seq_len
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(15,seq_len-8))
        self.pool1 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (1,4))
        self.pool2 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        #Getting the dimension after convolutions
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.name = 'var_net_'+str(seq_len)
        #Linear/Dense layers
        self.fc1 = nn.Linear(16*4, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,10)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10,2)
        self.dropout= nn.Dropout(0.4)
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                layer.zero_grad()
                
    def forward(self, x):
        #Conv -> ReLU -> MaxPool
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #Linear->ReLU->Dropou
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) #reshaping after convolution
        x = self.dropout(self.bn1(F.relu(self.fc1(x)))) 
        x = self.dropout(self.bn2(F.relu(self.fc2(x)))) 
        x = self.dropout(F.relu(self.fc3(x)))
        predictions = x.argmax(1)
        probabilities = x.softmax(1)

        return x, predictions, probabilities
    
    
class variable_atchley_net(torch.nn.Module):
    """
    PyTorch version of DeepCAT CNN
    seq_len represents the length (# of AAs) of the CDR3 region (L = 12, 13, ..., 16)
    When initializing CNN_CDR3, the L should be specified 
    """ 
    def __init__(self, seq_len):
        super(variable_atchley_net, self).__init__()
        #Convolutions
        self.length = seq_len
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(10,seq_len-8))
        self.pool1 = nn.MaxPool2d(kernel_size = (1,4), stride=(1,1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (10,2))
        self.pool2 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        #Getting the dimension after convolutions
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.name = 'var_atchley_net_'+str(seq_len)
        #Linear/Dense layers
        #self.fc1 = nn.Linear(16*4*2, 10)
        self.fc1 = nn.Linear(16*4*2, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,10)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10,2)
        self.dropout= nn.Dropout(0.4)
        
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                layer.zero_grad()
                
    def forward(self, x):
        #Conv -> ReLU -> MaxPool
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #Linear->ReLU->Dropou
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) #reshaping after convolution
        x = self.dropout(self.bn1(F.relu(self.fc1(x)))) 
        x = self.dropout(self.bn2(F.relu(self.fc2(x)))) 
        x = self.dropout(F.relu(self.fc3(x)))
        
        predictions = x.argmax(1)
        probabilities = x.softmax(1)

        return x, predictions, probabilities
class xd(torch.nn.Module):
    """
    PyTorch version of DeepCAT CNN
    seq_len represents the length (# of AAs) of the CDR3 region (L = 12, 13, ..., 16)
    When initializing CNN_CDR3, the L should be specified 
    """ 
    def __init__(self, seq_len):
        super(xd, self).__init__()
        #Convolutions
        self.length = seq_len
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5,4))
        self.pool1 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (1,2))
        self.pool2 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        #Getting the dimension after convolutions
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.name = 'xd_'+str(seq_len)
        #Linear/Dense layers
        #self.fc1 = nn.Linear(16*4*2, 10)
        self.fc1 = nn.Linear(16*(seq_len-6), 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,10)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10,2)
        self.dropout= nn.Dropout(0.4)
            
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                layer.zero_grad()
                
    def forward(self, x):
        #Conv -> ReLU -> MaxPool
        #print("in", x.shape)
        x = F.relu(self.conv1(x))
        #print("after conv1",x.shape)
        x = self.pool1(x)
        #print("after pool1", x.shape)
        x = F.relu(self.conv2(x))
        #print("after conv2", x.shape)
        x = self.pool2(x)
        #print("after pool2", x.shape)
        #Linear->ReLU->Dropou
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) #reshaping after convolution
        #print("after reshape", x.shape)
        x = self.dropout(self.bn1(F.relu(self.fc1(x)))) 
        x = self.dropout(self.bn2(F.relu(self.fc2(x)))) 
        x = self.dropout(F.relu(self.fc3(x)))
        
        predictions = x.argmax(1)
        probabilities = x.softmax(1)

        return x, predictions, probabilities

class xd2(torch.nn.Module):
    """
    PyTorch version of DeepCAT CNN
    seq_len represents the length (# of AAs) of the CDR3 region (L = 12, 13, ..., 16)
    When initializing CNN_CDR3, the L should be specified 
    """ 
    def __init__(self, seq_len):
        super(xd2, self).__init__()
        #Convolutions
        self.length = seq_len
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5,4))
        self.pool1 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (1,2))
        self.pool2 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        #Getting the dimension after convolutions
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.name = 'xd2_'+str(seq_len)
        #Linear/Dense layers
        if seq_len >= 14 :
            self.fc1 = nn.Sequential(nn.Linear(16*(seq_len-6), 96),
                                     nn.BatchNorm1d(96),
                                     nn.relu(),
                                     nn.Linear(96,50))
        else :
            self.fc1 = nn.Linear(16*(seq_len-6), 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,10)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10,2)
        self.dropout= nn.Dropout(0.4)
            
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                layer.zero_grad()
                
    def forward(self, x):
        #Conv -> ReLU -> MaxPool
        #print("in", x.shape)
        x = F.relu(self.conv1(x))
        #print("after conv1",x.shape)
        x = self.pool1(x)
        #print("after pool1", x.shape)
        x = F.relu(self.conv2(x))
        #print("after conv2", x.shape)
        x = self.pool2(x)
        #print("after pool2", x.shape)
        #Linear->ReLU->Dropou
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) #reshaping after convolution
        #print("after reshape", x.shape)
        x = self.dropout(self.bn1(F.relu(self.fc1(x)))) 
        x = self.dropout(self.bn2(F.relu(self.fc2(x)))) 
        x = self.dropout(F.relu(self.fc3(x)))
        
        predictions = x.argmax(1)
        probabilities = x.softmax(1)

        return x, predictions, probabilities
    
class cropped_og(torch.nn.Module):
    """
    testing cropped sequences
    """ 
    def __init__(self, seq_len):
        super(cropped_og, self).__init__()
        #Convolutions
        self.length = seq_len-6
        self.nodes = seq_len-10
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(10,2))
        self.pool1 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        self.conv2 = nn.Conv2d(20, 40, kernel_size = (10,2))
        self.pool2 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        #Getting the dimension after convolutions
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.name = 'cropped_og_'+str(seq_len)
        #Linear/Dense layers
        #self.fc1 = nn.Linear(16*4*2, 10)
        self.fc1 = nn.Linear(40*self.nodes*2, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,10)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10,2)
        self.dropout= nn.Dropout(0.4)
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                layer.zero_grad()
                
    def forward(self, x):
        #Conv -> ReLU -> MaxPool
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #Linear->ReLU->Dropou
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) #reshaping after convolution
        x = self.dropout(self.bn1(F.relu(self.fc1(x)))) 
        x = self.dropout(self.bn2(F.relu(self.fc2(x)))) 
        x = self.dropout(F.relu(self.fc3(x)))
        
        predictions = x.argmax(1)
        probabilities = x.softmax(1)

        return x, predictions, probabilities
    
class cropped_atchley(torch.nn.Module):
    """
    testing cropped sequences
    """ 
    def __init__(self, seq_len):
        super(cropped_atchley, self).__init__()
        #Convolutions
        self.length = seq_len-6
        self.nodes = seq_len-10
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5,2))
        self.pool1 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        self.conv2 = nn.Conv2d(20, 40, kernel_size = (1,2))
        self.pool2 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        #Getting the dimension after convolutions
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.name = 'cropped_atchley1_'+str(seq_len)
        #Linear/Dense layers
        #self.fc1 = nn.Linear(16*4*2, 10)
        self.fc1 = nn.Linear(40*self.nodes, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,10)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10,2)
        self.dropout= nn.Dropout(0.4)
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                layer.zero_grad()
                
    def forward(self, x):
        #Conv -> ReLU -> MaxPool
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        #Linear->ReLU->Dropou
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) #reshaping after convolution
        x = self.dropout(self.bn1(F.leaky_relu(self.fc1(x)))) 
        x = self.dropout(self.bn2(F.leaky_relu(self.fc2(x)))) 
        x = self.dropout(F.relu(self.fc3(x)))
        
        predictions = x.argmax(1)
        probabilities = x.softmax(1)

        return x, predictions, probabilities
    
# MOD MAPPING DICT to create models
MOD_MAPPING = {'deepcat':deepcat_cnn, 
               'richie' : richie_net,#To be renamed
               'variable': variable_net,
               'variable_atchley' : variable_atchley_net,
               'cropped_1' : cropped_atchley,
               'cropped_og' : cropped_og,
               'xd' : xd,
               'xd2' : xd2
              }
               
#PATTERN MAPPING DICT to load filenames of models
PATTERN_MAPPING = {'deepcat': 'deepcat_cnn_', 
                   'richie' : 'richie_net_',#To be renamed
                   'variable': 'var_net_',
                   'variable_atchley' : 'var_atchley_net_',
                   'cropped_1' : 'cropped_atchley1_',
                   'cropped_og' : 'cropped_og_',
                   'xd' : 'xd_',
                   'xd2' : 'xd2_'
                  }

########### HELPERS ###########
def get_models(keys=[12,13,14,15,16], arch = 'richie'):
    model_dict = {}
    for key in keys:
        model = MOD_MAPPING[arch](key) 
        model_dict[key]=model
        del model
    return model_dict

def load_models(PATH, keys=[12,13,14,15,16], arch = 'richie', eval_=False):
    model_dict = get_models(keys, arch)
    files = os.listdir(PATH)
    files = [f for f in files if '.pth.tar' in f]
    
    for key in keys:
        pattern = PATTERN_MAPPING[arch]+str(key)
        chkpt = {}
        for fn in files:
            if pattern in fn:
                name = os.path.join(PATH,fn)
                chkpt = torch.load(name)
            else:continue
        model_dict[key].load_state_dict(chkpt['state_dict'])
        if eval_: model_dict[key].eval()
    print("Models & weights loaded")
    return model_dict

def model_info(PATH, keys = [12,13,14,15,16], arch='richie'):
    model_dict = get_models(keys, arch)
    files = os.listdir(PATH)
    files = [f for f in files if '.pth.tar' in f]
    
    for key in keys:
        pattern = PATTERN_MAPPING[arch]+str(key)
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
