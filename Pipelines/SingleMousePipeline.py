import numpy as np
import matplotlib.pyplot as plt
#from EnsemblePursuitPyTorch_threshold import EnsemblePursuitPyTorch
import sys
sys.path.append("..")
from EnsemblePursuitModule.EnsemblePursuitPyTorch import EnsemblePursuitPyTorch
from EnsemblePursuitModule.EnsemblePursuitNumpy import EnsemblePursuitNumpy
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
    def __init__(self,data_path, mouse_filename,model,nr_of_components,lambd_=None,save=False):
        self.data_path=data_path
        self.model=model
        self.lambd_=lambd_
        self.nr_of_components=nr_of_components
        self.mouse_filename=mouse_filename

    def fit_model(self):
        data = io.loadmat(self.data_path+self.mouse_filename)
        resp = data['stim'][0]['resp'][0]
        spont =data['stim'][0]['spont'][0]
        if self.model=='EnsemblePursuit_numpy':
            X=subtract_spont(spont,resp).T
            options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
            ep_np=EnsemblePursuitNumpy(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
            start=time.time()
            U,V=ep_np.fit_transform(X)
            end=time.time()
            tm=end-start
            print('Time', tm)
            if save=True
                np.save(self.save_path+filename+'_V_ep_numpy.npy',V)
                np.save(self.save_path+filename+'_U_ep_numpy.npy',U)
            return U,V
