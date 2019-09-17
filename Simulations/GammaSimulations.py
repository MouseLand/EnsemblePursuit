import numpy as np
import sys
sys.path.append("..")
from EnsemblePursuitModule.EnsemblePursuitPyTorch import EnsemblePursuitPyTorch
from EnsemblePursuitModule.EnsemblePursuitNumpy import EnsemblePursuitNumpy
import matplotlib.pyplot as plt
import seaborn as sns
from NMF import fit_NMF

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

    def V_corr_mat(self,model_string):
        corrs_V_original=np.zeros((self.V.shape[0],self.V.shape[0]))
        corrs_V_fitted=np.zeros((self.V.shape[0],self.V.shape[0]))
        corrs_V_orig_fit=np.zeros((self.V.shape[0],self.V.shape[0]))
        for j in range(0,self.V.shape[0]):
            for i in range(0,self.V.shape[0]):
                corrs_V_orig_fit[j,i]=np.corrcoef(self.V_orig[j,:],self.V[i,:])[0,1]
                if i!=j:
                    corrs_V_original[j,i]=np.corrcoef(self.V_orig[j,:],self.V_orig[i,:])[0,1]
                    corrs_V_fitted[j,i]=np.corrcoef(self.V[j,:],self.V[i,:])[0,1]
        max_corrs_orig=np.max(corrs_V_original,axis=0)
        max_corrs_fitted=np.max(corrs_V_fitted,axis=0)
        max_corrs_orig_fit=np.max(corrs_V_orig_fit,axis=0)
        plt.figure(figsize=(15,8))
        plt.subplot(231)
        ax1=plt.subplot(2,3,1)
        sns.heatmap(corrs_V_original,ax=ax1,vmin=0, vmax=1)
        plt.title('V_original corrs')
        ax2=plt.subplot(2,3,2)
        sns.heatmap(corrs_V_fitted,ax=ax2,vmin=0, vmax=1)
        plt.title('V_fitted corrs')
        ax3=plt.subplot(2,3,3)
        sns.heatmap(corrs_V_orig_fit,ax=ax3,vmin=0, vmax=1)
        plt.title('Orig vs fitted corrs')
        ax4=plt.subplot(2,3,4)
        ax4.hist(max_corrs_orig)
        plt.axvline(np.median(max_corrs_orig),color='r')
        ax4.text(0.55,0.8,'Median '+str(np.median(max_corrs_orig))[0:4],transform=ax4.transAxes)
        plt.title('Max corrs V_original')
        ax5=plt.subplot(2,3,5)
        print(max_corrs_fitted)
        print(max_corrs_orig_fit)
        ax5.hist(max_corrs_fitted)
        plt.axvline(np.median(max_corrs_fitted),color='r')
        if model_string=='ICA':
            ax5.text(0.55,0.8,'Median '+str(np.median(max_corrs_fitted))[0:3]+'e-15',transform=ax5.transAxes)
        if model_string=='PCA':
            ax5.text(0.55,0.8,'Median '+str(np.median(max_corrs_fitted))[0:3]+'e-16',transform=ax5.transAxes)
        else:
            ax5.text(0.55,0.8,'Median '+str(np.median(max_corrs_fitted))[0:4],transform=ax5.transAxes)
        plt.title('Max corrs V_fitted')
        ax6=plt.subplot(2,3,6)
        ax6.hist(max_corrs_orig_fit)
        plt.title('Best correlation of fitted ensemble with an original')
        plt.axvline(np.median(max_corrs_orig_fit),color='r')
        ax6.text(0.55,0.8,'Median '+str(np.median(max_corrs_orig_fit))[0:4],transform=ax6.transAxes)
        plt.subplots_adjust(wspace=0.6,hspace=0.5)
        plt.show()


    def run_and_fit(self,model_string,nr_components,nr_timepoints,nr_neurons,lambd=0):
        np.random.seed(7)
        X=self.simulate_data(nr_components,nr_timepoints,nr_neurons)
        #X=self.simulate_data_w_noise(nr_components,nr_timepoints,nr_neurons,noise_ampl_mult=4)
        if model_string=='EnsemblePursuit':
            options_dict={'seed_neuron_av_nr':10,'min_assembly_size':1}
            ep_pt=EnsemblePursuitPyTorch(n_ensembles=nr_components,lambd=lambd,options_dict=options_dict)
            U,V=ep_pt.fit_transform(X)
            self.U=U.numpy()
            self.V=V.numpy().T
        if model_string=='EnsemblePursuitNumpy':
            options_dict={'seed_neuron_av_nr':10,'min_assembly_size':1}
            ep_np=EnsemblePursuitNumpy(n_ensembles=nr_components,lambd=lambd,options_dict=options_dict)
            U,V =ep_np.fit_transform(X)
            self.U=U
            self.V=V.T
        if model_string=='NMF':
            self.U,self.V=fit_NMF(X,nr_components,n_epoch=1000)



    def variance_explained_across_neurons(self):
        '''
        From sklearn:
        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
        ((y_true - y_pred) ** 2).sum() and v is the total sum of squares
        ((y_true - y_true.mean()) ** 2).sum().
        '''
        X=self.U_orig@self.V_orig
        approx=self.U@self.V
        u=[]
        v=[]
        for j in range(X.shape[0]):
            u_j=((X[j,:]-approx[j,:])**2).sum()
            v_j=((X[j,:]-np.mean(X[j,:]))**2).sum()
            u.append(u_j)
            v.append(v_j)
        u=np.array(u)
        v=np.array(v)
        plt.plot(-np.divide(u,v)+1)
        plt.ylim(0,1)
        plt.title('Variance explained across neurons')
        plt.show()
        print('Total variance explained, averaged over neurons is:',(1-np.mean(u)/np.mean(v)))



    #def variance_explained_across_components(self,U,V):

    def sparsity(self,model_string,nr_of_components):
        prop_lst=[]
        prop_lst_orig=[]
        for j in range(0,nr_of_components):
            #Set small numbers to zero
            self.U[self.U<0.000001]=0
            if model_string=='EnsemblePursuitNumpy' or 'NMF':
                proportion_of_nonzeros=np.sum(self.U[:,j]!=0)
                proportion_of_nonzeros_orig=np.sum(self.U_orig[:,j]!=0)
            else:
                proportion_of_nonzeros=np.sum(self.U[j,:]!=0)
                proportion_of_nonzero_orig=np.sum(self.U_orig[j,:]!=0)
            prop_lst.append(proportion_of_nonzeros)
            prop_lst_orig.append(proportion_of_nonzeros_orig)
            #matplotlib.rcParams.update({'font.size': 22})
        print('Mean proportion of nonzeros in learned U:', np.mean(prop_lst)/self.U.shape[0])
        print('Mean proportion of nonzeros in simulated U:',np.mean(prop_lst_orig)/self.U.shape[0])
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
