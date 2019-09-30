import numpy as np
import matplotlib.pyplot as plt
#from EnsemblePursuitPyTorch import EnsemblePursuitPyTorch
from EnsemblePursuitPyTorchFast import EnsemblePursuitPyTorchFast
from EnsemblePursuitNumpyFast import EnsemblePursuitNumpyFast
#from EnsemblePursuitNumpy import EnsemblePursuitNumpy
from EnsemblePursuitRegressionInit2 import EnsemblePursuitRegressionInit
from EnsemblePursuitVarianceInit2 import EnsemblePursuitVarianceInit
from EnsemblePursuitThresholdInit import EnsemblePursuitThresholdInit
from EnsemblePursuitTFIDFInit import EnsemblePursuitTFIDF
from utils import test_train_split, evaluate_model_torch, subtract_spont, corrcoef, PCA
import pandas as pd
from scipy import io
import time
import glob
import os

class ModelPipeline():
    def __init__(self,data_path, save_path, model,nr_of_components,lambd_=None, alpha=None):
        self.data_path=data_path
        self.save_path=save_path
        self.model=model
        self.lambd_=lambd_
        self.alpha=alpha
        self.nr_of_components=nr_of_components
        self.mat_file_lst=['natimg2800_M170717_MP034_2017-09-11.mat','natimg2800_M160825_MP027_2016-12-14.mat','natimg2800_M161025_MP030_2017-05-29.mat','natimg2800_M170604_MP031_2017-06-28.mat','natimg2800_M170714_MP032_2017-09-14.mat','natimg2800_M170714_MP032_2017-08-07.mat','natimg2800_M170717_MP033_2017-08-20.mat']


    def fit_model(self):
        for filename in self.mat_file_lst:
            print(filename)
            data = io.loadmat(self.data_path+filename)
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
                np.save(self.save_path+filename+'_V_ep_numpy.npy',V)
                np.save(self.save_path+filename+'_U_ep_numpy.npy',U)
                np.save(self.save_path+filename+'_timing_ep_numpy.npy',tm)
            if self.model=='EnsemblePursuit_pytorch':
                X=subtract_spont(spont,resp).T
                options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
                ep_pt=EnsemblePursuitPyTorch(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
                start=time.time()
                U,V=ep_pt.fit_transform(X)
                end=time.time()
                tm=end-start
                print('Time', tm)
                np.save(self.save_path+filename+'_V_ep_pytorch.npy',V)
                np.save(self.save_path+filename+'_U_ep_pytorch.npy',U)
                np.save(self.save_path+filename+'_timing_ep_pytorch.npy',tm)
            if self.model=='EnsemblePursuit_numpy_fast':
                X=subtract_spont(spont,resp).T
                options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
                ep_np=EnsemblePursuitNumpyFast(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
                start=time.time()
                U,V=ep_np.fit_transform(X)
                end=time.time()
                tm=end-start
                print('Time', tm)
                np.save(self.save_path+filename+'_V_ep_numpy_fast.npy',V)
                np.save(self.save_path+filename+'_U_ep_numpy_fast.npy',U)
                np.save(self.save_path+filename+'_timing_ep_numpy_fast.npy',tm)
            if self.model=='EnsemblePursuit_pytorch_fast':
                X=subtract_spont(spont,resp).T
                options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
                ep_pt=EnsemblePursuitPyTorchFast(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
                start=time.time()
                U,V=ep_pt.fit_transform(X)
                end=time.time()
                tm=end-start
                print('Time', tm)
                np.save(self.save_path+filename+'_V_ep_pytorch_fast.npy',V)
                np.save(self.save_path+filename+'_U_ep_pytorch_fast.npy',U)
                np.save(self.save_path+filename+'_timing_ep_pytorch_fast.npy',tm)
            if self.model=='EnsemblePursuit_reg':
                X=subtract_spont(spont,resp).T
                options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
                ep_pt=EnsemblePursuitRegressionInit(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
                start=time.time()
                U,V=ep_pt.fit_transform(X)
                end=time.time()
                tm=end-start
                print('Time', tm)
                np.save(self.save_path+filename+'_V_ep_pytorch_reg.npy',V)
                np.save(self.save_path+filename+'_U_ep_pytorch_reg.npy',U)
                np.save(self.save_path+filename+'_timing_ep_pytorch_reg.npy',tm)

            if self.model=='EnsemblePursuit_var':
                X=subtract_spont(spont,resp).T
                options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
                ep_pt=EnsemblePursuitVarianceInit(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
                start=time.time()
                U,V=ep_pt.fit_transform(X)
                end=time.time()
                tm=end-start
                print('Time', tm)
                np.save(self.save_path+filename+'_V_ep_pytorch_var.npy',V)
                np.save(self.save_path+filename+'_U_ep_pytorch_var.npy',U)
                np.save(self.save_path+filename+'_timing_ep_pytorch_var.npy',tm)

            if self.model=='EnsemblePursuit_thresh':
                X=subtract_spont(spont,resp).T
                options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
                ep_pt=EnsemblePursuitThresholdInit(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
                start=time.time()
                U,V=ep_pt.fit_transform(X)
                end=time.time()
                tm=end-start
                print('Time', tm)
                np.save(self.save_path+filename+'_V_ep_pytorch_thresh.npy',V)
                np.save(self.save_path+filename+'_U_ep_pytorch_thresh.npy',U)
                np.save(self.save_path+filename+'_timing_ep_pytorch_thresh.npy',tm)
            if self.model=='EnsemblePursuit_tfidf':
                X=subtract_spont(spont,resp).T
                options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
                ep_pt=EnsemblePursuitTFIDF(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
                start=time.time()
                U,V=ep_pt.fit_transform(X)
                end=time.time()
                tm=end-start
                print('Time', tm)
                np.save(self.save_path+filename+'_V_ep_pytorch_tfidf.npy',V)
                np.save(self.save_path+filename+'_U_ep_pytorch_tfidf.npy',U)
                np.save(self.save_path+filename+'_timing_ep_pytorch_tfidf.npy',tm)

    def knn(self):
       if self.model=='SparsePCA':
             model_string='*V_sPCA.npy'
       if self.model=='EnsemblePursuit_numpy':
             model_string='*_V_ep_numpy.npy'
       if self.model=='EnsemblePursuit_pytorch':
             model_string='*_V_ep_pytorch.npy'
       if self.model=='EnsemblePursuit_numpy_fast':
             model_string='*_V_ep_numpy_fast.npy'
       if self.model=='EnsemblePursuit_pytorch_fast':
             model_string='*_V_ep_pytorch_fast.npy'
       if self.model=='EnsemblePursuit_reg':
             model_string='*V_ep_pytorch_reg.npy'
       if self.model=='EnsemblePursuit_var':
             model_string='*V_ep_pytorch_var.npy'
       if self.model=='EnsemblePursuit_thresh':
             model_string='*V_ep_pytorch_thresh.npy'
       if self.model=='EnsemblePursuit_tfidf':
             model_string='*V_ep_pytorch_tfidf.npy'
       if self.model=='NMF':
             model_string='*_V_NMF.npy'
       if self.model=='PCA':
             model_string='*_V_pca.npy'
       if self.model=='LDA':
             model_string='*_V_lda.npy'
       if self.model=='all':
             #self.save_path=self.data_path
             model_string='*.mat'
       columns=['Experiment','accuracy']
       acc_df=pd.DataFrame(columns=columns)
       print(self.save_path)
       for filename in glob.glob(os.path.join(self.save_path, model_string)):
             V=np.load(filename)
             istim_path=filename[len(self.save_path):len(self.save_path)+len(self.mat_file_lst[0])]
             istim=io.loadmat(self.data_path+istim_path)['stim']['istim'][0][0].astype(np.int32)
             istim -= 1 # get out of MATLAB convention
             istim = istim[:,0]
             nimg = istim.max() # these are blank stims (exclude them)
             V = V[istim<nimg, :]
             istim = istim[istim<nimg]
             x_train,x_test,y_train,y_test=test_train_split(V,istim)
             acc=evaluate_model_torch(x_train,x_test)
             acc_df=acc_df.append({'Experiment':filename[len(self.save_path):],'accuracy':acc},ignore_index=True)
       pd.options.display.max_colwidth = 300
       print(acc_df)
       print(acc_df.describe())
       return acc_df     
        

