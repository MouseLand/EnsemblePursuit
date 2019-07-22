import numpy as np
import time


class EnsemblePursuitNumpy():
    def __init__(self,n_ensembles,lambd,options_dict):
        self.n_ensembles=n_ensembles
        self.lambd=lambd
        self.options_dict=options_dict

    def simulate_data(self,nr_components,nr_timepoints,nr_neurons):
        zeros_for_U=np.random.choice([0,1], nr_neurons*nr_components, p=[0.75, 0.25]).reshape((nr_neurons,nr_components))
        U=np.random.normal(loc=2,scale=1,size=(nr_neurons,nr_components))
        U=np.abs(U*zeros_for_U)
        V=np.random.normal(loc=0,scale=1,size=(nr_components,nr_timepoints))
        X=U@V
        X=self.zscore(X)
        return V
        
    def zscore(self,X):
        mean_stimuli=np.mean(X.T,axis=0)
        std_stimuli=np.std(X.T,axis=0,ddof=1)+1e-10
        X=np.subtract(X.T,mean_stimuli)
        X=np.divide(X,std_stimuli)
        return X.T

    def calculate_cost_delta(self,C_summed,current_v):
        cost_delta=np.clip(C_summed,a_min=0,a_max=None)**2/(self.sz[1]*(current_v**2).sum())-self.lambd
        return cost_delta
    
    def mask_cost_delta(self,selected_neurons,cost_delta):
        mask=np.zeros((selected_neurons.shape[0]),dtype=bool)
        mask[selected_neurons==0]=1
        mask[selected_neurons!=0]=0
        masked_cost_delta=mask*cost_delta
        return masked_cost_delta

    def sum_C(self,C_summed_unnorm,C,max_delta_neuron):
        C_summed_unnorm=C_summed_unnorm+C[:,max_delta_neuron]
        return C_summed_unnorm

    def sum_v(self, v, max_delta_neuron, X):
        current_v=v+X[max_delta_neuron,:]
        return current_v

    def fit_one_ensemble(self,X):
        start=time.time()
        C=X@X.T
        end=time.time()
        #A parameter to account for how many top neurons we sample from. It starts from 1,
        #because we choose the top neuron when possible, e.g. when we can find an ensemble
        # that is larger than min ensemble size. If there is no ensemble with the top neuron
        # we increase the number of neurons to sample from.
        self.n_neurons_for_sampling=1
        n=0
        min_assembly_size=self.options_dict['min_assembly_size']
        safety_it=0
        #A while loop for trying sampling other neurons if the found ensemble size is smaller
        #than threshold.
        neuron_sublst=[]
        while n<min_assembly_size:
            if self.n_neurons_for_sampling>1:
                 seed=self.repeated_seed(C)
            else:
                 seed=self.sorting_for_seed(C)
            n=1
            current_v=X[seed,:]
            current_v_unnorm=current_v.copy()
            selected_neurons=np.zeros((X.shape[0]),dtype=bool)
            neuron_sublst.append(seed)
            #Seed current_v
            selected_neurons[seed]=1
            #Fake cost to initiate while loop
            max_cost_delta=1000
            C_summed_unnorm=0
            max_delta_neuron=seed
            corr_lst=[]
            while max_cost_delta>0:
                #Add the x corresponding to the max delta neuron to C_sum. Saves computational 
                #time.
                C_summed_unnorm=self.sum_C(C_summed_unnorm,C,max_delta_neuron)
                C_summed=(1./n)*C_summed_unnorm
                cost_delta=self.calculate_cost_delta(C_summed,current_v)
                #invert the 0's and 1's in the array which stores which neurons have already 
                #been selected into the assembly to use it as a mask
                masked_cost_delta=self.mask_cost_delta(selected_neurons,cost_delta)
                max_cost_delta=np.max(masked_cost_delta)
                max_delta_neuron=np.argmax(masked_cost_delta)
                if max_cost_delta>0:
                    neuron_sublst.append(max_delta_neuron)
                    selected_neurons[max_delta_neuron]=1
                    current_v_unnorm= self.sum_v(current_v_unnorm,max_delta_neuron,X)
                    n+=1
                    current_v=(1./n)*current_v_unnorm
                    corr_sublst=[]
                    for v in range(0,self.target.shape[0]):
                        corr=np.corrcoef(self.target[v,:],current_v)[0,1]
                        corr_sublst.append(corr)
                    corr_lst.append(max(corr_sublst))
            safety_it+=1
            #Increase number of neurons to sample from if while loop hasn't been finding any assemblies.     
            if safety_it>0:
                self.n_neurons_for_sampling=50
            if safety_it>50:
                self.n_neurons_for_sampling=100
            if safety_it>100:
                self.n_neurons_for_sampling=500
            if safety_it>600:
                self.n_neurons_for_sampling=1000
            if safety_it>1600:
                raise ValueError('Assembly capacity too big, can\'t fit model')
        self.neuron_lst.append(neuron_sublst)
        self.corrs.append(corr_lst)
        current_u=np.zeros((X.shape[0],1))
        current_u[selected_neurons,0]=np.clip(C_summed[selected_neurons],a_min=0,a_max=None)/(current_v**2).sum()
        self.U=np.concatenate((self.U,current_u),axis=1)
        self.V=np.concatenate((self.V,current_v.reshape(1,self.sz[1])),axis=0)
        return current_u, current_v
    
    def repeated_seed(self,C):
        nr_neurons_to_av=self.options_dict['seed_neuron_av_nr']
        sorted_similarities=np.sort(C,axis=1)[:,:-1][:,self.sz[0]-nr_neurons_to_av-1:] 
        average_similarities=np.mean(sorted_similarities,axis=1)
        top_neurons=np.argsort(average_similarities)
        sample_top_neuron=np.random.randint(0,self.n_neurons_for_sampling,size=1)
        top_neurons=top_neurons[self.sz[0]-(self.n_neurons_for_sampling):]
        seed=top_neurons[sample_top_neuron][0]
        return seed

    def sorting_for_seed(self,C):
        '''
        This function sorts the similarity matrix C to find neurons that are most correlated
        to their nr_neurons_to_av neighbors (we average over the neighbors).
        '''
        nr_neurons_to_av=self.options_dict['seed_neuron_av_nr']
        sorted_similarities=np.sort(C,axis=1)[:,:-1][:,self.sz[0]-nr_neurons_to_av-1:] 
        average_similarities=np.mean(sorted_similarities,axis=1)
        seed=np.argmax(average_similarities)
        return seed
   
    def fit_transform(self,X):
        np.random.seed(7)
        self.target=self.simulate_data(self.n_ensembles,X.shape[1],X.shape[0])
        self.neuron_lst=[]
        self.corrs=[]
        X=self.zscore(X)
        self.sz=X.shape
        self.U=np.zeros((X.shape[0],1))
        self.V=np.zeros((1,X.shape[1]))
        for iteration in range(0,self.n_ensembles):
            current_u, current_v=self.fit_one_ensemble(X)
            U_V=current_u.reshape(self.sz[0],1)@current_v.reshape(1,self.sz[1])
            X=X-U_V
            print('ensemble nr', iteration)
            cost=np.mean(X*X)
            print('cost',cost) 
        #After fitting arrays discard the zero initialization rows and columns from U and V.
        self.U=self.U[:,1:]
        self.V=self.V[1:,:]
        return self.U, self.V.T, self.corrs
