import numpy as np
import torch
import sys
sys.path.append("..")
from scipy import stats
from scipy import io
import time
import sys
sys.path.append("..")
from EnsemblePursuitModule.EnsemblePursuitNumpyFast import EnsemblePursuitNumpyFast
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import FastICA

class ReceptiveFieldsPlusBehaviorPipeline():
    def __init__(self,data_path,model,nr_of_components):
        self.data_path=data_path
        self.model=model
        self.nr_of_components=nr_of_components

    def fit(self):
        dt=1
        spks= np.load(self.data_path+'spks.npy')
        print('Shape of the data matrix, neurons by timepoints:',spks.shape)
        iframe = np.load(self.data_path+'iframe.npy') # iframe[n] is the microscope frame for the image frame n
        ivalid = iframe+dt<spks.shape[-1] # remove timepoints outside the valid time range
        iframe = iframe[ivalid]
        S = spks[:, iframe+dt]
        del spks
        S = stats.zscore(S, axis=1) # z-score the neural activity before doing anything
        if self.model=='EnsemblePursuit':
            options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
            nr_of_components=200
            ep_np=EnsemblePursuitNumpyFast(n_ensembles=self.nr_of_components,lambd=0.01,options_dict=options_dict)
            U,V=ep_np.fit_transform(S)
            return U,V
        if self.model=='ICA':
            ica=FastICA(n_components=self.nr_of_components)
            U=ica.fit_transform(S.T)
            V=ica.components_.T
            return U,V

    def train_test_split(self,NT):
        nsegs = 20
        nt=NT
        nlen  = nt/nsegs
        ninds = np.linspace(0,nt-nlen,nsegs).astype(int)
        itest = (ninds[:,np.newaxis] + np.arange(0,nlen*0.25,1,int)).flatten()
        itrain = np.ones(nt, np.bool)
        itrain[itest] = 0
        return itrain, itest

    def fit_receptive_field_on_minus_off(self,V):
        spks= np.load(self.data_path+'spks.npy')
        iframe = np.load(self.data_path+'iframe.npy')
        mov    = np.load(self.data_path+'mov.npy')# these are the visual stimuli shown
        dt = 1 # time offset between stimulus presentation and response
        ivalid = iframe+dt<spks.shape[-1] # remove timepoints outside the valid time range
        iframe = iframe[ivalid]
        mov = mov[:, :, ivalid]
        Sp = V.T
        Sp = zscore(Sp, axis=1)
        ly, lx, nstim = mov.shape
        del spks
        NT = Sp.shape[1]
        NN=Sp.shape[0]
        print(NT)
        itrain,itest=self.train_test_split(NT)

        X = np.reshape(mov, [-1, NT]) # reshape to Npixels by Ntimepoints
        X = X-0.5 # subtract the background
        #X = np.abs(X) # does not matter if a pixel is black (0) or white (1)
        X = stats.zscore(X, axis=1)/NT**.5  # z-score each pixel separately
        npix = X.shape[0]

        lam = 0.1
        ncomps = Sp.shape[0]
        B0 = np.linalg.solve((X[:,itrain] @ X[:,itrain].T + lam * np.eye(npix)),  (X[:,itrain] @ Sp[:,itrain].T)) # get the receptive fields for each neuron

        B0 = np.reshape(B0, (ly, lx, ncomps))
        B0 = gaussian_filter(B0, [.5, .5, 0]) # smooth each receptive field a little

        plt.figure(figsize=(18, 8))
        rfmax = np.max(B0)
        for j in range(200):
            plt.subplot(10,20,j+1)
            rf = B0[:,:,j]
            # rfmax = np.max(np.abs(rf))
            plt.imshow(rf, aspect='auto', cmap = 'bwr', vmin = -rfmax, vmax = rfmax) # plot the receptive field for each neuron
            #plt.title('PC %d'%(1+j))
            plt.axis('off')

        plt.show()

        Spred = np.reshape(B0, (-1,NN)).T @ X[:,itest]
        varexp = 1.0 - Spred.var(axis=1)/Sp[:,itest].var(axis=1)
        print(varexp)
        print(Spred.shape)
        print(Sp.shape)
        original=Sp[:,itest]
        print(original.shape)
        plt.plot(Spred[0,:100],label='Spred')
        plt.plot(original[0,:100],label='Sp')
        plt.legend()
        corr_lst=[]
        for j in range(0,200):
            corr_lst.append(np.corrcoef(Sp[j,itest],Spred[j,:])[0,1])
        print(corr_lst)
        return B0,corr_lst

    def fit_receptive_field_on_plus_off(self,V):
        spks= np.load(self.data_path+'spks.npy')
        iframe = np.load(self.data_path+'iframe.npy')
        mov    = np.load(self.data_path+'mov.npy')# these are the visual stimuli shown
        dt = 1 # time offset between stimulus presentation and response
        ivalid = iframe+dt<spks.shape[-1] # remove timepoints outside the valid time range
        iframe = iframe[ivalid]
        mov = mov[:, :, ivalid]
        Sp = V.T
        Sp = zscore(Sp, axis=1)
        ly, lx, nstim = mov.shape
        del spks
        NT = Sp.shape[1]
        NN=Sp.shape[0]
        print(NT)
        itrain,itest=self.train_test_split(NT)

        X = np.reshape(mov, [-1, NT]) # reshape to Npixels by Ntimepoints
        X = X-0.5 # subtract the background
        X = np.abs(X) # does not matter if a pixel is black (0) or white (1)
        X = stats.zscore(X, axis=1)/NT**.5  # z-score each pixel separately
        npix = X.shape[0]

        lam = .001
        ncomps = Sp.shape[0]
        B0 = np.linalg.solve((X[:,itrain] @ X[:,itrain].T + lam * np.eye(npix)),  (X[:,itrain] @ Sp[:,itrain].T)) # get the receptive fields for each neuron

        B0 = np.reshape(B0, (ly, lx, ncomps))
        B0 = gaussian_filter(B0, [.5, .5, 0]) # smooth each receptive field a little

        plt.figure(figsize=(18, 8))
        rfmax = np.max(B0)
        for j in range(200):
            plt.subplot(10,20,j+1)
            rf = B0[:,:,j]
            # rfmax = np.max(np.abs(rf))
            plt.imshow(rf, aspect='auto', cmap = 'bwr', vmin = -rfmax, vmax = rfmax) # plot the receptive field for each neuron
            #plt.title('PC %d'%(1+j))
            plt.axis('off')

        plt.show()

        Spred = np.reshape(B0, (-1,NN)).T @  X[:,itest]
        varexp = 1.0 - (Spred**2).mean(axis=-1)

        corr_lst=[]
        for j in range(0,self.nr_of_components):
            corr_lst.append(np.corrcoef(Sp[j,itest],Spred[j,:])[0,1])
        print(corr_lst)
        return B0, corr_lst

    def combined_plot(self,B_minus,B_plus):
        plt.figure(figsize=(40, 20))
        rfmax_minus = np.max(B_minus)
        rfmax_plus=np.max(B_plus)
        ly, lx, ncomps=B_minus.shape
        combined=np.zeros((ly,lx,400))
        for j in range(0,400,40):
            combined[:,:,j:(j+20)]=B_minus[:,:,j//2:(j//2+20)]
            combined[:,:,(j+20):(j+40)]=B_plus[:,:,j//2:(j//2+20)]
        for j in range(400):
            plt.subplot(20,20,j+1)
            rf = combined[:,:,j]
            # rfmax = np.max(np.abs(rf))
            if j//2==0:
                rfmax=rfmax_minus
            elif j//2==1:
                rfmax=rfmax_plus
            plt.imshow(rf, aspect='auto', cmap = 'bwr', vmin = -rfmax, vmax = rfmax) # plot the receptive field for each neuron
            #plt.title('PC %d'%(1+j))
            plt.axis('off')

        plt.show()

    def fit_to_behavior(self,V):
        iframe = np.load(self.data_path+'iframe.npy')
        spks= np.load(self.data_path+'spks.npy')
        dt = 1 # time offset between stimulus presentation and response
        ivalid = iframe+dt<spks.shape[-1] # remove timepoints outside the valid time range
        iframe = iframe[ivalid]
        del spks
        proc = np.load(self.data_path+'cam1_TX39_2019_05_31_1_proc_resampled.npy', allow_pickle=True).item()
        motSVD = proc['motSVD'][:,iframe+dt]
        motSVD -= motSVD.mean(axis=1)[:,np.newaxis]
        beh=motSVD
        NT = motSVD.shape[1]
        itrain,itest=self.train_test_split(NT)
        covM = np.matmul(beh[:,itrain], beh[:,itrain].T)
        lam = 1e5 # regularizer
        covM += lam*np.eye(beh.shape[0])
        A = np.linalg.solve(covM, np.matmul(beh[:,itrain], V.T[:,itrain].T))
        print(beh.shape)
        Vpred = np.matmul(A.T, beh[:,itest])
        print(Vpred.shape)
        varexp = 1 - ((Vpred - V.T[:,itest])**2).sum(axis=1)/(V.T[:,itest]**2).sum(axis=1)
        corr_lst=[]
        for j in range(0,self.nr_of_components):
            corr_lst.append(np.corrcoef(Vpred[j,:],V.T[j,itest])[0,1])
        return corr_lst

    def selectivity_scatter_plot(self,corr_lst_stim,corr_lst_beh):
        plt.scatter(corr_lst_stim,corr_lst_beh)
        plt.xlabel('Correlations from stimulus predictions on test set')
        plt.ylabel('Correlations from behavior predictions on test set')
        plt.show()

    def behavior_correlation_seed(self):
        iframe = np.load(self.data_path+'iframe.npy')
        spks= np.load(self.data_path+'spks.npy')
        dt = 1 # time offset between stimulus presentation and response
        ivalid = iframe+dt<spks.shape[-1] # remove timepoints outside the valid time range
        iframe = iframe[ivalid]
        S = spks[:, iframe+dt]
        S = stats.zscore(S, axis=1) # z-score the neural activity before doing anything
        del spks
        proc = np.load(self.data_path+'cam1_TX39_2019_05_31_1_proc_resampled.npy', allow_pickle=True).item()
        motSVD = proc['motSVD'][:,iframe+dt]
        options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
        ep_np=EnsemblePursuitNumpyFast(n_ensembles=1,lambd=0.01,options_dict=options_dict)
        cells=ep_np.fit_one_ensemble_suite2p(S,motSVD[0,:])
        print(np.corrcoef(np.mean(S[cells,:],axis=0),motSVD[0,:]))
