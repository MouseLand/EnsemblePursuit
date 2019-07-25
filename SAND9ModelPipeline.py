import numpy as np
import matplotlib.pyplot as plt
from EnsemblePursuitPyTorch import EnsemblePursuitPyTorch
from EnsemblePursuitNumpy import EnsemblePursuitNumpy
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

class ModelPipeline():
    def __init__(self,data_path, save_path, model,nr_of_components,lambd_=None):
        self.data_path=data_path
        self.save_path=save_path
        self.model=model
        self.lambd_=lambd_
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
            if self.model=='SparsePCA':
                X=subtract_spont(spont,resp)
                X=zscore(X)
                sPCA=SparsePCA(n_components=self.nr_of_components,random_state=7, max_iter=100, n_jobs=-1,verbose=1)
                start=time.time()
                model=sPCA.fit(X)
                end=time.time()
                elapsed_time=end-start
                U=model.components_
                V=sPCA.transform(X)
                np.save(self.save_path+filename+'_U_sPCA.npy',U)
                np.save(self.save_path+filename+'_V_sPCA.npy',V)
                np.save(self.save_path+filename+'_time_sPCA.npy',elapsed_time)
            if self.model=='ICA':
                X=subtract_spont(spont,resp)
                X=zscore(X)
                ICA=FastICA(n_components=self.nr_of_components,random_state=7)
                start=time.time()
                V=ICA.fit_transform(X)
                end=time.time()
                elapsed_time=end-start
                U=ICA.components_
                np.save(self.save_path+filename+'_U_ICA.npy',U)
                np.save(self.save_path+filename+'_V_ICA.npy',V)
                np.save(self.save_path+filename+'_time_ICA.npy',elapsed_time)
         
    def knn(self):
       if self.model=='SparsePCA':
             model_string='*V_sPCA.npy'
       if self.model=='ICA':
             model_string='*_V_ICA.npy'
       if self.model=='EnsemblePursuit_numpy':
             model_string='*_V_ep_numpy.npy'
       if self.model=='EnsemblePursuit_pytorch':
             model_string='*_V_ep_pytorch.npy'
       if self.model=='NMF':
             model_string='*_V_NMF.npy'
       if self.model=='PCA':
             model_string='*_V_pca.npy'
       if self.model=='LDA':
             model_string='*_V_lda.npy'
       if self.model=='NMF_regularization_experiments':
            model_string='*_V_NMF_regularization_experiments.npy'
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
             if self.model!='NMF_regularization_experiments':
                 acc_df=acc_df.append({'Experiment':filename[len(self.save_path):],'accuracy':acc},ignore_index=True)
             if self.model=='NMF_regularization_experiments':
                 acc_df=acc_df.append({'Experiment':filename[len(self.save_path):],'accuracy':acc, 'alpha':filename[:-19][-1:]},ignore_index=True)
       grouped=acc_df.groupby(['alpha']).mean()
       print(grouped)
       print(grouped.describe())
       pd.options.display.max_colwidth = 300
       print(acc_df)
       print(acc_df.describe())
       return acc_df     

    def fit_ridge(self):
        images=io.loadmat(self.data_path+'images/images_natimg2800_all.mat')['imgs']
        images=images.transpose((2,0,1))
        images=images.reshape((2800,68*270))
        from utils import PCA
        reduced_images=PCA(images)
        if self.model=='EnsemblePursuit_pytorch':
            model_string='*V_ep_pytorch.npy'
        if self.model=='SparsePCA':
            model_string='*V_sPCA.npy'
        if self.model=='ICA':
            model_string='*_V_ICA.npy'
        if self.model=='NMF':
            model_string='*_V_NMF.npy'
        if self.model=='PCA':
            model_string='*V_pca.npy'
        if self.model=='LDA':
            model_string='*_V_lda.npy'
        for filename in glob.glob(os.path.join(self.save_path, model_string)):
            print(filename)
            istim_path=filename[len(self.save_path):len(self.save_path)+len(self.mat_file_lst[0])]
            stim=io.loadmat(self.data_path+istim_path)['stim']['istim'][0][0]
            #test train split
            components=np.load(filename)
            x_train,x_test,y_train,y_test=test_train_split(components,stim)
            y_train=y_train-1
            reduced_images_=reduced_images[y_train]
            for alpha in [5000]:
                assembly_array=[] 
                for assembly in range(0,self.nr_of_components):
                    av_resp=(x_train[:,assembly].T+x_test[:,assembly].T)/2
                    reg=ridge_regression(reduced_images_,av_resp,alpha=alpha)
                    assembly_array.append(reg)
                assembly_array=np.array(assembly_array) 
                if self.model=='EnsemblePursuit_pytorch':
                    file_string=self.save_path+istim_path+'_ep_pytorch_reg.npy'
                if self.model=='SparsePCA':      
                    file_string=self.save_path+istim_path+'_sPCA_reg.npy'
                if self.model=='ICA':
                    file_string=self.save_path+istim_path+'_ica_reg.npy'
                if self.model=='NMF':
                    file_string=self.save_path+istim_path+'_NMF_reg.npy'
                if self.model=='PCA':
                    file_string=self.save_path+istim_path+'_pca_reg.npy'
                if self.model=='LDA':
                    file_string=self.save_path+istim_path+'_lda_reg.npy'
                np.save(file_string,assembly_array)

    def plot_all_receptive_fields(self):
        if self.model=='SparsePCA':
            model_string='*sPCA_reg.npy'
        if self.model=='EnsemblePursuit_pytorch':
            model_string='*ep_pytorch_reg.npy'
        if self.model=='NMF':
            model_string='*_NMF_reg.npy'
        if self.model=='PCA':
            model_string='*_pca_reg.npy'
        if self.model=='LDA':
            model_string='*_lda_reg.npy'
        if self.model=='ICA':
            model_string='*_ica_reg.npy'
        for filename in glob.glob(os.path.join(self.save_path, model_string)):
            print(filename)
            assembly_array=np.load(filename)
            assembly_array=assembly_array.reshape(10,15,18360)
            fig=plt.figure(figsize=(10,15))
            ax=[]
            i=0
            for ind1 in range(0,10):
                for ind2 in range(0,15):
                    ax=fig.add_axes([ind1/10,ind2/15,1./10,1./15])
                    ax.imshow(assembly_array[ind1,ind2,:].reshape(68,270),cmap=plt.get_cmap('bwr'))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(x=ind1/10,y=ind2/15,s=str(i))
                    i+=1
            plt.show()

    def NMF_regul_exps_fit(self):
        self.model='NMF_regularization_experiments'
        alphas=[0.01,0.1,1,10,100]
        powers=[-2,-1,0,1,2]
        ind_dict={0.01:0,0.1:1,1:3,10:4,100:5}
        for filename in self.mat_file_lst:
            data = io.loadmat(self.data_path+filename)
            resp = data['stim'][0]['resp'][0]
            spont =data['stim'][0]['spont'][0]
            X=subtract_spont(spont,resp)
            X-=X.min(axis=0)
            for alpha in alphas:
                model = NMF(n_components=self.nr_of_components, init='nndsvd', random_state=7,alpha=alpha,l1_ratio=1.0)
                start=time.time()
                V=model.fit_transform(X)
                end=time.time()
                time_=end-start
                print(end-start)
                U=model.components_
                np.save(self.save_path+filename+'_'+str(ind_dict[alpha])+'_U_NMF_reg_exps.npy',U)
                np.save(self.save_path+filename+'_'+str(ind_dict[alpha])+'_V_NMF_reg_exps.npy',V)
                


        

