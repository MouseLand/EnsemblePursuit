import numpy as np


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
            print(seed)
            n=100

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
        #print(X)
        self.fit_one_ensemble(X)
