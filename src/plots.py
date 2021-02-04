from models import *
from preprocessing import *
from train_eval_helpers import *
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import mpl.pyplot as plt
mpl.rcParams['figure.dpi']= 150
sns.set_style('darkgrid')
#sns.set_palette("coolwarm", n_colors=2)

def plot_loss(train, val):
    