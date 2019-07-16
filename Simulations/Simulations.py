from EnsemblePursuitSimulations import EnsemblePursuitPyTorch
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import seaborn as sns

class Simulations():
    def simulate_data(self,nr_components,nr_timepoints,nr_neurons):
        zeros_for_U=np.random.choice([0,1], nr_neurons*nr_components, p=[0.75, 0.25]).reshape((nr_neurons,nr_components))
        U=np.random.normal(loc=2,scale=1,size=(nr_neurons,nr_components))
        U=np.abs(U*zeros_for_U)
        V=np.random.normal(loc=0,scale=1,size=(nr_components,nr_timepoints))
        X=U@V
        X=zscore(X,axis=1)
        self.U_orig=U
        self.V_orig=V
        return X

    def fit_to_simulation(self,model_string,nr_components,nr_timepoints,nr_neurons,X,lambd=0):
        options_dict={'seed_neuron_av_nr':10,'min_assembly_size':8}
        if model_string=='EnsemblePursuit':
            ep_pt=EnsemblePursuitPyTorch(n_ensembles=nr_components,lambd=lambd,options_dict=options_dict)
            U,V=ep_pt.fit_transform(X)
            U=U.numpy()
            V=V.numpy().T
        if model_string=='ICA':
            print('boyaka')
            ica=FastICA(n_components=nr_components,random_state=7)
            V=ica.fit_transform(X).T
            U=ica.mixing_
        print(U.shape)
        print(V.shape)
        return U,V

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
        fig = plt.figure(1,figsize=(15,8))
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
        ax5.hist(max_corrs_fitted)
        plt.axvline(np.median(max_corrs_fitted),color='r')
        if model_string=='ICA':
            ax5.text(0.55,0.8,'Median '+str(np.median(max_corrs_fitted))[0:3]+'e-15',transform=ax5.transAxes)
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
        
    
    def run_and_fit(self,model_string,nr_components,nr_timepoints,nr_neurons,lambd=0):
        np.random.seed(7)
        X=self.simulate_data(nr_components,nr_timepoints,nr_neurons)
        if model_string=='EnsemblePursuit':
            self.U,self.V=self.fit_to_simulation('EnsemblePursuit',nr_components,nr_timepoints,nr_neurons,X,lambd)
        if model_string=='ICA':
           self.U,self.V=self.fit_to_simulation('ICA',nr_components,nr_timepoints,nr_neurons,X.T)
        print('SHPS', self.U.shape, self.V.shape)
        self.orig=X
        self.approx=self.U@self.V
        print('orig',self.orig.shape)
        print('approx',self.approx.shape)

