import numpy as np
import matplotlib.pyplot as plt
#from EnsemblePursuitPyTorch_threshold import EnsemblePursuitPyTorch
import sys
sys.path.append("..")
from EnsemblePursuitModule.EnsemblePursuitPyTorch import EnsemblePursuitPyTorch
from EnsemblePursuitModule.EnsemblePursuitNumpy import EnsemblePursuitNumpy
#from EnsemblePursuitNumpy import EnsemblePursuitNumpy
from EnsemblePursuitModule.EnsemblePursuitNumpyVarExp import EnsemblePursuitNumpyVarExp
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
    def __init__(self,data_path, mouse_filename,model,nr_of_components,lambd_=None,save=False,save_path=None):
        self.data_path=data_path
        self.model=model
        self.lambd_=lambd_
        self.nr_of_components=nr_of_components
        self.mouse_filename=mouse_filename
        self.save=save
        self.save_path=save_path

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
            if self.save==True:
                np.save(self.save_path+self.mouse_filename+'_V_ep_numpy.npy',V)
                np.save(self.save_path+self.mouse_filename+'_U_ep_numpy.npy',U)
            return U,V
        if self.model=='EnsemblePursuit_numpy_var_exp':
            X=subtract_spont(spont,resp).T
            options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
            ep_np=EnsemblePursuitNumpyVarExp(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
            start=time.time()
            U,V=ep_np.fit_transform(X)
            end=time.time()
            tm=end-start
            print('Time', tm)
            if self.save==True:
                np.save(self.save_path+self.mouse_filename+'_V_ep_numpy.npy',V)
                np.save(self.save_path+self.mouse_filename+'_U_ep_numpy.npy',U)
            return U,V

    def knn(self,V):
        columns=['Experiment','accuracy']
        acc_df=pd.DataFrame(columns=columns)
        istim=io.loadmat(self.data_path+self.mouse_filename)['stim']['istim'][0][0].astype(np.int32)
        istim -= 1 # get out of MATLAB convention
        istim = istim[:,0]
        nimg = istim.max() # these are blank stims (exclude them)
        V = V[istim<nimg, :]
        istim = istim[istim<nimg]
        x_train,x_test,y_train,y_test=test_train_split(V,istim)
        acc=evaluate_model_torch(x_train,x_test)
        acc_df=acc_df.append({'Experiment':self.mouse_filename,'accuracy':acc},ignore_index=True)
        pd.options.display.max_colwidth = 300
        print(acc_df)
        return acc_df

    def cross_validation(self):
        data = io.loadmat(self.data_path+self.mouse_filename)
        resp = data['stim'][0]['resp'][0]
        spont =data['stim'][0]['spont'][0]
        X=subtract_spont(spont,resp).T
        istim=io.loadmat(self.data_path+self.mouse_filename)['stim']['istim'][0][0].astype(np.int32)
        istim -= 1 # get out of MATLAB convention
        istim = istim[:,0]
        nimg = istim.max()
        X=X[:,istim<nimg]
        istim = istim[istim<nimg]
        print(X.shape)
        print(istim.shape)
        x_train,x_test,y_train,y_test=test_train_split(X.T,istim)
        print(x_train.shape)
        if self.model=='EnsemblePursuit_numpy_var_exp':
            options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
            ep_train=EnsemblePursuitNumpyVarExp(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
            start=time.time()
            U_train,V_train=ep_train.fit_transform(x_train.T)
            end=time.time()
            tm=end-start
            print('Time', tm)
            bundle={'U':U_train,'V':V_train,'mouse_name':self.mouse_filename,'time':tm}
            if self.save==True:
                np.save(self.save_path+self.mouse_filename+'_ep_varexp_train.npy',bundle)
            options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
            ep_test=EnsemblePursuitNumpyVarExp(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
            start=time.time()
            U_test,V_test=ep_train.fit_transform(x_test.T)
            end=time.time()
            tm=end-start
            print('Time', tm)
            bundle={'U':U_test,'V':V_test,'mouse_name':self.mouse_filename,'time':tm}
            if self.save==True:
                np.save(self.save_path+self.mouse_filename+'_ep_varexp_test.npy',bundle)
        if self.model=='EnsemblePursuit_numpy':
            options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
            ep_train=EnsemblePursuitNumpy(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
            start=time.time()
            U_train,V_train=ep_train.fit_transform(x_train.T)
            end=time.time()
            tm=end-start
            print('Time', tm)
            bundle={'U':U_train,'V':V_train,'mouse_name':self.mouse_filename,'time':tm}
            if self.save==True:
                np.save(self.save_path+self.mouse_filename+'_ep_numpy_train.npy',bundle)
            options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
            ep_test=EnsemblePursuitNumpy(n_ensembles=self.nr_of_components,lambd=self.lambd_,options_dict=options_dict)
            start=time.time()
            U_test,V_test=ep_train.fit_transform(x_test.T)
            end=time.time()
            tm=end-start
            print('Time', tm)
            bundle={'U':U_test,'V':V_test,'mouse_name':self.mouse_filename,'time':tm}
            if self.save==True:
                np.save(self.save_path+self.mouse_filename+'_ep_numpy_test.npy',bundle)



    def fit_ridge(self,V):
        images=io.loadmat(self.data_path+'images/images_natimg2800_all.mat')['imgs']
        images=images.transpose((2,0,1))
        images=images.reshape((2800,68*270))
        from utils import PCA
        reduced_images=PCA(images)
        stim=io.loadmat(self.data_path+self.mouse_filename)['stim']['istim'][0][0].astype(np.int32)
        x_train,x_test,y_train,y_test=test_train_split(V,stim)
        y_train=y_train-1
        reduced_images_=reduced_images[y_train]
        for alpha in [5000]:
            assembly_array=[]
            for assembly in range(0,self.nr_of_components):
                av_resp=(x_train[:,assembly].T+x_test[:,assembly].T)/2
                reg=ridge_regression(reduced_images_,av_resp,alpha=alpha)
                assembly_array.append(reg)
            assembly_array=np.array(assembly_array)
            if self.save==True:
                if self.model=='EnsemblePursuit_numpy':
                    file_string=self.save_path+self.mouse_filename+'_ep_numpy_'+str(alpha)+'reg.npy'
                if self.model=='EnsemblePursuit_pytorch':
                    file_string=self.save_path+self.mouse_filename+'_ep_pytorch_'+str(alpha)+'reg.npy'
                np.save(file_string)
        return assembly_array

    def plot_all_receptive_fields(self,assembly_array):
        first_dim=10
        second_dim=self.nr_of_components//10
        assembly_array=assembly_array.reshape(first_dim,second_dim,18360)
        fig=plt.figure(figsize=(first_dim,second_dim))
        ax=[]
        i=0
        for ind1 in range(0,first_dim):
            for ind2 in range(0,second_dim):
                ax=fig.add_axes([ind1/first_dim,ind2/second_dim,1./first_dim,1./second_dim])
                ax.imshow(assembly_array[ind1,ind2,:].reshape(68,270),cmap=plt.get_cmap('bwr'))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(x=ind1/first_dim,y=ind2/second_dim,s=str(i))
                i+=1
        plt.show()

    def variance_explained_across_neurons(self,U,V):
        '''
        From sklearn:
        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
        ((y_true - y_pred) ** 2).sum() and v is the total sum of squares
        ((y_true - y_true.mean()) ** 2).sum().
        '''
        #Fetch the original data and convert it into the same form as what goes into the
        #matrix factorization model
        data = io.loadmat(self.data_path+self.mouse_filename)
        resp = data['stim'][0]['resp'][0]
        spont =data['stim'][0]['spont'][0]
        X=subtract_spont(spont,resp).T
        X=zscore(X.T).T
        u=[]
        v=[]
        approx=U@V.T
        for j in range(X.shape[0]):
            u_j=((X[j,:]-approx[j,:])**2).sum()
            v_j=((X[j,:]-np.mean(X[j,:]))**2).sum()
            u.append(u_j)
            v.append(v_j)
        u=np.array(u)
        v=np.array(v)
        plt.plot(-np.divide(u,v)+1)
        plt.title('Variance explained across neurons')
        plt.show()
        print('Total variance explained, averaged over neurons is:',(1-np.mean(u)/np.mean(v)))



    #def variance_explained_across_components(self,U,V):

    def sparsity(self,U):
        prop_lst=[]
        for j in range(0,self.nr_of_components):
            #Set small numbers to zero
            U[U<0.000001]=0
            if self.model=='EnsemblePursuit_pytorch' or self.model=='EnsemblePursuit_numpy' or self.model=='EnsemblePursuit_adaptive' or self.model=='EnsemblePursuit_numpy_var_exp':
                proportion_of_nonzeros=np.sum(U[:,j]!=0)
            else:
                proportion_of_nonzeros=np.sum(U[j,:]!=0)
            prop_lst.append(proportion_of_nonzeros)
            #matplotlib.rcParams.update({'font.size': 22})
        plt.plot(prop_lst,'o')
        plt.show()
        fig=plt.figure(figsize=(6,6))
        ax=fig.add_subplot(111)
        ax.semilogy(range(1,len(prop_lst)+1), prop_lst,'o')
        ax.set_xlabel('component order')
        ax.set_ylabel('number of neurons')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('w')
        plt.show()
