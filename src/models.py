import torch
import torch.nn as nn
import torch.nn.functional as F 
import os 

########### HELPERS ###########
def get_models(keys, arch = 'deepcat'):
    model_dict = {}
    for key in keys:
        if arch == 'deepcat':
            model = deepcat_cnn(key)
        elif arch == 'richie':
            model = richie_net(key)
        model_dict[key]=model
        del model
    return model_dict

def load_models(PATH, keys=[12,13,14,15,16], arch = 'deepcat'):
    model_dict = get_models(keys, arch)
    files = os.listdir(PATH)
    files = [f for f in files if '.pth.tar' in f]
    
    for key in keys:
        if arch =='deepcat':pattern = 'deepcat_cnn_'+str(key)
        else: pattern='richie_net_'+str(key)
        chkpt = {}
        for fn in files:
            if pattern in fn:
                name = os.path.join(PATH,fn)
                chkpt = torch.load(name)
            else:continue
        model_dict[key].load_state_dict(chkpt['state_dict'])
    print("Models & weights loaded")
    return model_dict

def model_info(PATH, keys = [12,13,14,15,16], arch='deepcat'):
    model_dict = get_models(keys)
    files = os.listdir(PATH)
    files = [f for f in files if '.pth.tar' in f]
    
    for key in keys:
        if arch =='deepcat':pattern = 'deepcat_cnn_'+str(key)
        else: pattern='richie_net_'+str(key)
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

#class richie_net(torch.nn.Module):
#    """
#    PyTorch version of DeepCAT CNN
#    seq_len represents the length (# of AAs) of the CDR3 region (L = 12, 13, ..., 16)
#    When initializing CNN_CDR3, the L should be specified 
#    """ 
#    def __init__(self, seq_len):
#        super(richie_net, self).__init__()
#        #Convolutions
#        self.conv1 = nn.Conv2d(1,8, kernel_size = (10,3))
#        self.pool1 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,1))
#        self.conv2 = nn.Conv2d(8,16, kernel_size = (10,3))
#        self.pool2 = nn.MaxPool2d(kernel_size= (1,2), stride=(1,1))
#        #Getting the dimension after convolutions
#        self.dummy_param = nn.Parameter(torch.empty(0))
#        self.length = seq_len
#        self.name = 'richie_net_'+str(seq_len)
#        #Linear/Dense layers
#        self.fc1 = nn.Linear(16*(self.length-6)*2, 50)
#        self.bn1 = nn.BatchNorm1d(50)
#        self.fc2 = nn.Linear(50,10)
#        self.bn2 = nn.BatchNorm1d(10)
#        self.fc3 = nn.Linear(10,2)
#        self.dropout= nn.Dropout(0.4)
#        
#    def reset_parameters(self):
#        for layer in self.children():
#            if hasattr(layer, 'reset_parameters'):
#                layer.reset_parameters()
#                layer.zero_grad()
#                
#    def forward(self, x):
#        #Conv -> ReLU -> MaxPool
#        x = F.leaky_relu(self.conv1(x))
#        x = self.pool1(x)
#        x = F.leaky_relu(self.conv2(x))
#        x = self.pool2(x)
#        #Linear->ReLU->Dropout
#        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) #reshaping after convolution
#        #FC layers
#        x = self.dropout(self.bn1(F.leaky_relu(self.fc1(x)))) 
#        x = self.dropout(self.bn2(F.leaky_relu(self.fc2(x)))) #Getting binary logits
#        x = self.dropout(F.leaky_relu(self.fc3(x)))
#        predictions = x.argmax(1)
#        probabilities = x.softmax(1)
#
#        return x, predictions, probabilities
    
class richie_net(torch.nn.Module):
    """
    RICHIE SIMPLER
    """ 
    def __init__(self, seq_len):
        super(richie_net, self).__init__()
        #Convolutions
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(20,2))
        self.pool1 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (1,2))
        self.pool2 = nn.MaxPool2d(kernel_size = (1,2), stride=(1,1))
        #Getting the dimension after convolutions
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.length = seq_len
        self.name = 'richie_net_'+str(seq_len)
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
        #Linear->ReLU->Dropout
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) #reshaping after convolution
        #FC layers
        x = self.dropout((F.relu(self.fc1(x)))) 
        x = self.dropout((F.relu(self.fc2(x)))) #Getting binary logits
        #x = self.dropout(F.relu(self.fc3(x)))
        predictions = x.argmax(1)
        probabilities = x.softmax(1)

        return x, predictions, probabilities