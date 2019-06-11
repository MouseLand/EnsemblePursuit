import numpy as np
import time


class EnsemblePursuitNumpy():
    def __init__(self,n_ensembles,lambd,options_dict):
        self.n_ensembles=n_ensembles
        self.lambd=lambd
        self.options_dict=options_dict

        
    def zscore(self,X):
        mean_stimuli=np.mean(X,axis=1)[...,np.newaxis]
        std_stimuli=np.std(X,axis=1)[...,np.newaxis]+0.0000000001
        X=np.subtract(X,mean_stimuli)
        X=np.divide(X,std_stimuli)
        return X

    def calculate_cost_delta(self,C_summed,current_v):
        cost_delta=np.clip(C_summed,a_min=0,a_max=None)**2/(self.sz[1]*(current_v**2).sum())-self.lambd
        return cost_delta
    
    def mask_cost_delta(self,selected_neurons,cost_delta):
        mask=np.zeros((selected_neurons.shape[0]),dtype=bool)
        mask[selected_neurons==0]=1
        mask[selected_neurons!=0]=0
        masked_cost_delta=mask*cost_delta
        return masked_cost_delta

    def sum_C(self,C_summed,C,max_delta_neuron):
        C_summed=C_summed+C[:,max_delta_neuron]
        return C_summed

    def sum_v(self, v, max_delta_neuron, X):
        current_v=v+X[max_delta_neuron,:]
        return current_v

    def fit_one_ensemble(self,X):
        C=X@X.T
        #A parameter to account for how many top neurons we sample from. It starts from 1,
        #because we choose the top neuron when possible, e.g. when we can find an ensemble
        # that is larger than min ensemble size. If there is no ensemble with the top neuron
        # we increase the number of neurons to sample from.
        self.n_neurons_for_sampling=1
        top_neurons=self.sorting_for_seed(C)
        n=1
        min_assembly_size=self.options_dict['min_assembly_size']
        max_delta_cost=1000
        safety_it=0
        #A while loop for trying sampling other neurons if the found ensemble size is smaller
        #than threshold.
        while n<min_assembly_size:
            seed=self.sample_seed_neuron(top_neurons)
            n=1
            current_v=X[seed,:]
            current_v_unnorm=current_v
            selected_neurons=np.zeros((X.shape[0]),dtype=bool)
            #Seed current_v
            selected_neurons[seed]=1
            #Fake cost to initiate while loop
            max_cost_delta=1000
            C_summed_unnorm=0
            max_delta_neuron=seed
            while max_cost_delta>0:
                C_summed=self.sum_C(C_summed_unnorm,C,max_delta_neuron)
                C_summed_unnorm=C_summed.copy()
                C_summed=(1./n)*C_summed
                cost_delta=self.calculate_cost_delta(C_summed,current_v)
                #invert the 0's and 1's in the array which stores which neurons have already 
                #been selected into the assembly to use it as a mask
                masked_cost_delta=self.mask_cost_delta(selected_neurons,cost_delta)
                max_cost_delta=np.max(masked_cost_delta)
                max_delta_neuron=np.argmax(masked_cost_delta)
                if max_cost_delta>0:
                    selected_neurons[max_delta_neuron]=1
                    current_v_unnorm= self.sum_v(current_v_unnorm,max_delta_neuron,X)
                    n+=1
                    current_v=(1./n)*current_v_unnorm
            print(n)
            print('numpy current_v',current_v)
            n=100000000



    def sample_seed_neuron(self,top_neurons):
        sample_top_neuron=np.random.randint(self.n_neurons_for_sampling,size=1)
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
        print(sorted_similarities.shape)
        average_similarities=np.mean(sorted_similarities,axis=1)
        top_neurons=np.argsort(average_similarities)
        print('top neurons', top_neurons)
        return top_neurons
   
    def fit_transform(self,X):
        X=self.zscore(X)
        self.sz=X.shape
        self.fit_one_ensemble(X)
