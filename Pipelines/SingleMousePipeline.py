import numpy as np
import matplotlib.pyplot as plt
#from EnsemblePursuitPyTorch_threshold import EnsemblePursuitPyTorch
import sys
sys.path.append("..")
from EnsemblePursuitModule.EnsemblePursuitPyTorch import EnsemblePursuitPyTorch
#from EnsemblePursuitNumpy import EnsemblePursuitNumpy
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.linear_model import ridge_regression
from utils import test_train_split, evaluate_model_torch, subtract_spont, corrcoef, PCA,zscore
import pandas as pd
from scipy import io
import time
import glob
import os
from scipy import io
import matplotlib

class ModelPipelineSingleMouse():
    def __init__(self,data_path, save_path, model,nr_of_components,lambd_=None):
        self.data_path=data_path
        self.save_path=save_path
        self.model=model
        self.lambd_=lambd_
        self.nr_of_components=nr_of_components
