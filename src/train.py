"""
trains the 5 networks (or can choose which length to train [or not])

"""
#Allows relative imports, depends on where this script will end-up
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import shutil
import torch
from src.models import *
from src.preprocessing import *
from src.torch_util import *
from src.train_eval_helpers import *

def args_parser():
	parser = argparse.ArgumentParser(description='Trains the .')
	parser.add_argument('-i', type = str, default = os.,
                        help = 'relative path to input directory containing the tumor/normal, train&test CDR3 sequences. Format should be .txt, with no header. By default, it is ./TrainingData/, assuming train.py is located in the root of the github folder. ')
	parser.add_argument('-epochs', type=int, default = 300)
	return parser.parse_args()
