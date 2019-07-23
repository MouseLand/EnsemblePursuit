from EnsemblePursuitSimulations import EnsemblePursuitPyTorch
from EnsemblePursuitWithCorrReturn import EnsemblePursuitNumpy
#from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA, NMF, LatentDirichletAllocation

class Simulations():
    def zscore(self,X):
        X=X.T
        mean_stimuli=np.mean(X,axis=0)
        std_stimuli=np.std(X,axis=0,ddof=1)+1e-10
        X=np.subtract(X,mean_stimuli)
        X=np.divide(X,std_stimuli)
        return X.T

    def simulate_data(self,nr_components,nr_timepoints,nr_neurons):
        zeros_for_U=np.random.choice([0,1], nr_neurons*nr_components, p=[1-0.01, 0.01]).reshape((nr_neurons,nr_components))
        U=np.random.normal(loc=2,scale=1,size=(nr_neurons,nr_components))
        U=np.abs(U*zeros_for_U)
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
        zeros_for_U=np.random.choice([0,1], nr_neurons*nr_components, p=[1-0.01, 0.01]).reshape((nr_neurons,nr_components))
        U=np.random.normal(loc=2,scale=1,size=(nr_neurons,nr_components))
        U=np.abs(U*zeros_for_U)
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


    def find_nearest_U_comp(self,model_string,orig_ind):
        orig=self.U_orig[:,orig_ind]
        if model_string=='EnsemblePursuit':
            U=self.U[:,1:]
            dotpr=np.dot(orig.T,U)
            argmax=np.argmax(dotpr)+1
        else:
            U=self.U
            dotpr=np.dot(orig.T,U)
            argmax=np.argmax(dotpr)
        
        
        print('arg',argmax)
        plt.plot(dotpr)
        plt.show()
        #plt.plot(orig,label='Orig')
        plt.plot(self.U[:,argmax],label='Approx')
        plt.legend()
        plt.show()
        return argmax
    
    def find_and_plot_all_nearest_U(self):
        #print(self.U.shape)
        U=self.U[:,1:]
        match_lst=[]
        for i in range(0,self.U_orig.shape[1]):
            dotpr=np.dot(self.U_orig[:,i],U)
            argmax=np.argmax(dotpr)+1
            match_lst.append(argmax)
        print(match_lst)
        
    def find_nearest_V_comp(self,model_string):
        corrs=self.V_corr_mat(model_string)
        '''
        for orig_ind in range(0,corrs.shape[0]):
            most_correlated_v=np.argmax(corrs[orig_ind,:])
            plt.plot(self.V_orig[orig_ind,:])
            plt.plot(self.V[most_correlated_v,:])
            plt.show()
        '''
        nr_rows=str(int(np.ceil(corrs.shape[0]/5)))
        subplotnr=nr_rows+str(5)+str(corrs.shape[0])
        subplotnr=int(subplotnr)
        #fig = plt.figure(1,figsize=(15,8))
        fig=plt.figure(1,figsize=(15,1))
        for j in range(1,corrs.shape[0]+1):
            ax=fig.add_subplot(int(nr_rows),5,j)
            most_correlated_v=np.argmax(corrs[j-1,:])
            corr=np.max(corrs[j-1,:])
            ax.plot(self.V_orig[j-1,:])
            ax.plot(self.V[most_correlated_v,:])
            plt.title('Comp '+str(j)+', Corr '+str(corr)[0:4])
        plt.subplots_adjust(wspace=0.6,hspace=0.5)
        plt.show()
        return None
    
    def plot_nn(self,orig_ind,nr_time_points):
        approx=self.approx[orig_ind,:]
        orig=self.orig[orig_ind,:]
        plt.plot(range(0,nr_time_points),approx,label='Approximation')
        plt.plot(range(0,nr_time_points),orig,label='Original')
        plt.legend()
        plt.show()
        
    def plot_UV_cov(self,orig,fitted):
        return None
    
    def plot_recovered_U(self,nr_components):
        for j in range(0,nr_components):
            plt.plot(self.U[:,j])
        plt.xlabel('Neuron indices')
        plt.ylabel('Neuron weights')
        plt.title('Top 3 u\'s')
        plt.show()
        #print(self.U)
        
    def check_orthogonality(self):
        orth=self.V_orig.T@self.V_orig
        print('orth',orth)
        
    def neuron_corrs(self):
        corrs_neurons=np.zeros((self.orig.shape[0],self.orig.shape[0]))
        for j in range(0,self.orig.shape[0]):
            for i in range(0,self.orig.shape[0]):
                if j!=i:
                    corrs_neurons[j,i]=np.corrcoef(self.orig[j,:],self.orig[i,:])[0,1]
        sns.heatmap(corrs_neurons)
        plt.title('Neuron correlations')
        return None

    def V_corr_mat(self,model_string):
        corrs_V_original=np.zeros((self.V.shape[0],self.V.shape[0]))
        corrs_V_fitted=np.zeros((self.V.shape[0],self.V.shape[0]))
        corrs_V_orig_fit=np.zeros((self.V.shape[0],self.V.shape[0]))
        for j in range(0,self.V.shape[0]):
            for i in range(0,self.V.shape[0]):
                if j!=i:
                    corrs_V_original[j,i]=np.corrcoef(self.V_orig[j,:],self.V_orig[i,:])[0,1]
                    corrs_V_orig_fit[j,i]=np.corrcoef(self.V_orig[j,:],self.V[i,:])[0,1]
                    corrs_V_fitted[j,i]=np.corrcoef(self.V[j,:],self.V[i,:])[0,1]
        max_corrs_orig=np.max(corrs_V_original,axis=1)
        max_corrs_fitted=np.max(corrs_V_fitted,axis=1)
        max_corrs_orig_fit=np.max(corrs_V_orig_fit,axis=1)
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
        plt.title('Orig vs fitted max corrs')
        plt.axvline(np.median(max_corrs_orig_fit),color='r')
        ax6.text(0.55,0.8,'Median '+str(np.median(max_corrs_orig_fit))[0:4],transform=ax6.transAxes)
        plt.subplots_adjust(wspace=0.6,hspace=0.5)
        plt.show()

        return corrs_V_orig_fit

    def variance_explained_one_component(self,component_index):
        approx=(self.U[:,component_index].reshape(self.U.shape[0],1)@self.V[component_index,:].reshape(1,self.V.shape[1])).flatten()
        print(approx)
        explained_variance=((self.orig.flatten()-approx)**2).sum()
        normalizer=((self.orig.flatten()-np.mean(self.orig.flatten()))**2).sum()
        return 1-explained_variance/normalizer

    def total_variance_explained(self):
        explained_variance=((self.orig.flatten()-(self.U@self.V).flatten())**2).sum()
        normalizer=((self.orig.flatten()-np.mean(self.orig.flatten()))**2).sum()
        return 1-explained_variance/normalizer

    def variance_explained_by_neuron(self):
        UV=self.U@self.V
        print(UV.shape)
        print(self.orig.shape)
        vars_=[]
        for neuron in range(0,self.U.shape[0]):
            expl_var=((self.orig[neuron,:].flatten()-UV[neuron,:].flatten())**2).sum()
            normalizer=((self.orig[neuron,:].flatten()-np.mean(self.orig[neuron,:].flatten()))**2).sum()
            vars_.append(1-expl_var/normalizer)
        total_variance=np.mean(vars_)
        return total_variance, vars_

    def initial_component(self,nr_components,nr_timepoints,nr_neurons):
        X=self.simulate_data(nr_components,nr_timepoints,nr_neurons)
        n_neurons=[]
        median_corrs=[]
        rng=np.arange(0.01,0.5, 0.01)
        list_rng=list(rng)
        for param in list_rng:
            options_dict={'seed_neuron_av_nr':10,'min_assembly_size':1}
            ep_np=EnsemblePursuitNumpy(n_ensembles=nr_components,lambd=param,options_dict=options_dict)
            U,V=ep_np.fit_transform(X)
            self.U=U
            self.V=V.T
            n_neurons.append(np.count_nonzero(self.U[:,0].flatten()))
            corrs=self.V_corr_mat('EnsemblePursuitNumpy')
            max_corrs_orig_fit=np.max(corrs,axis=1)
            median_corrs.append(np.median(max_corrs_orig_fit))
        plt.plot(np.arange(0.01,0.5,0.01),n_neurons,label='Number of neurons')
        plt.legend()
        plt.show()
        plt.plot(np.arange(0.01,0.5,0.01),median_corrs,label='Median corr')
        plt.legend()
            
    def plot_corrs_as_neurons_added_(self):
        '''
        This function is buggy.
        '''
        print(self.neuron_lst)
        corrs_tot=[]
        for ensemble in self.neuron_lst:
            corrs=[]
            for neuron_ind in range(1,len(ensemble)+1):
                timeseries=self.orig[ensemble[:neuron_ind],:]
                av=np.mean(timeseries,axis=0)
                corrs_sublst=[]
                #print(self.V_orig.shape)
                for j in range(self.V_orig.shape[0]):
                    corr=np.corrcoef(self.V_orig[j,:],av)[0,1]
                    corrs_sublst.append(corr)
                corrs.append(max(corrs_sublst))
            corrs_tot.append(corrs)
        return corrs_tot

    def plot_corrs_as_neurons_added(self):
        nr_rows=str(int(np.ceil(len(self.corrs)/5)))
        subplotnr=nr_rows+str(5)+str(len(self.corrs))
        subplotnr=int(subplotnr)
        #fig = plt.figure(1,figsize=(15,8))
        fig=plt.figure(1,figsize=(10,4),dpi=50)
        for j in range(1,len(self.corrs)+1):
            ax=fig.add_subplot(int(nr_rows),5,j)
            ax.plot(self.corrs[j-1],'o')
            plt.title('Comp '+str(j))
        plt.subplots_adjust(wspace=0.6,hspace=0.5)
        plt.show()
                 
                
    
    def run_and_fit(self,model_string,nr_components,nr_timepoints,nr_neurons,lambd=0):
        np.random.seed(7)
        #X=self.simulate_data(nr_components,nr_timepoints,nr_neurons)
        X=self.simulate_data_w_noise(nr_components,nr_timepoints,nr_neurons,noise_ampl_mult=1)
        if model_string=='EnsemblePursuit':
            options_dict={'seed_neuron_av_nr':10,'min_assembly_size':1}


            ep_pt=EnsemblePursuitPyTorch(n_ensembles=nr_components,lambd=lambd,options_dict=options_dict)
            U,V=ep_pt.fit_transform(X)
            self.U=U.numpy()
            self.V=V.numpy().T
        if model_string=='EnsemblePursuitNumpy':
            options_dict={'seed_neuron_av_nr':10,'min_assembly_size':1}
            ep_np=EnsemblePursuitNumpy(n_ensembles=nr_components,lambd=lambd,options_dict=options_dict)
            U,V,self.corrs=ep_np.fit_transform(X)
            self.U=U
            self.V=V.T
        if model_string=='ICA':
           ica=FastICA(n_components=nr_components,random_state=7)
           self.V=ica.fit_transform(X.T).T
           self.U=ica.mixing_
        if model_string=='PCA':
           pca=PCA(n_components=nr_components,random_state=7)
           self.V=pca.fit_transform(X.T).T
           self.U=pca.components_.T
        if model_string=='sparsePCA':
           spca=SparsePCA(n_components=nr_components,random_state=7)
           self.V=spca.fit_transform(X.T).T
           self.U=spca.components_.T
        if model_string=='NMF':
           X-=X.min(axis=0)
           nmf=NMF(n_components=nr_components, init='nndsvd', random_state=7)
           self.V=nmf.fit_transform(X.T).T
           self.U=nmf.components_.T
        if model_string=='LDA':
           X-=X.min(axis=0)
           nmf=LatentDirichletAllocation(n_components=nr_components,random_state=7)
           self.V=nmf.fit_transform(X.T).T
           self.U=nmf.components_.T
        print('SHPS', self.U.shape, self.V.shape)
        self.orig=X
        self.approx=self.U@self.V
        print('orig',self.orig.shape)
        print('approx',self.approx.shape)

