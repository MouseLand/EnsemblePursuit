import numpy as np
import sys
sys.path.append("..")
from EnsemblePursuitModule.EnsemblePursuitPyTorch import EnsemblePursuitPyTorch
from EnsemblePursuitModule.EnsemblePursuitNumpy import EnsemblePursuitNumpy
import matplotlib.pyplot as plt

class GammaSimulations():
    def zscore(self,X):
        X=X.T
        mean_stimuli=np.mean(X,axis=0)
        std_stimuli=np.std(X,axis=0,ddof=1)+1e-10
        X=np.subtract(X,mean_stimuli)
        X=np.divide(X,std_stimuli)
        return X.T

    def simulate_data(self,nr_components,nr_timepoints,nr_neurons):
        k=5
        theta=1./5
        zeros_for_U=np.random.choice([0,1], nr_neurons*nr_components, p=[1-0.01, 0.01]).reshape((nr_neurons,nr_components))
        U=np.random.gamma(shape=k,scale=theta,size=(nr_neurons,nr_components))
        U=U*zeros_for_U
        V=np.random.normal(loc=0,scale=1,size=(nr_components,nr_timepoints))
        X=U@V
        X=self.zscore(X)
        self.U_orig=U
        self.V_orig=V
        plt.hist(np.sum(zeros_for_U,axis=0))
        plt.show()
        #print('Mean',X.shape,np.mean(X,axis=1))
        return X

    def simulate_data_w_noise(self,nr_components,nr_timepoints,nr_neurons, noise_ampl_mult):
        k=5
        theta=1./5
        zeros_for_U=np.random.choice([0,1], nr_neurons*nr_components, p=[1-0.01, 0.01]).reshape((nr_neurons,nr_components))
        U=np.random.gamma(shape=k,scale=theta,size=(nr_neurons,nr_components))
        U=U*zeros_for_U
        V=np.random.normal(loc=1,scale=1,size=(nr_components,nr_timepoints))
        X=U@V
        low_rank_std=np.std(X,axis=1)
        X_noisy=np.zeros((nr_neurons,nr_timepoints))
        for neuron in range(0,X.shape[0]):
            noise=np.random.normal(loc=0,scale=noise_ampl_mult*low_rank_std[neuron],size=(1,nr_timepoints))
            X_noisy[neuron,:]=X[neuron,:]+noise
        X=self.zscore(X_noisy)
        self.U_orig=U
        self.V_orig=V
        plt.hist(np.sum(zeros_for_U,axis=0))
        plt.show()
        #print('Mean',X.shape,np.mean(X,axis=1))
        return X
