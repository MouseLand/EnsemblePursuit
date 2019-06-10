import torch

class EnsemblePursuitPyTorch():
    def __init__(self, n_ensembles, lambd, options_dict):
        self.n_ensembles=n_ensembles
        self.lambd=lambd
        self.options_dict=options_dict
    
    def zscore(self,X):
        #Have to transpose X to make torch.sub and div work. Transpose back into 
        #original shape when done with calculations. 
        mean_stimuli=X.t().mean(dim=0)
        std_stimuli=X.t().std(dim=0)+0.0000000001
        
        X=torch.sub(X.t(),mean_stimuli)
        X=X.div(std_stimuli)
        return X.t()

    def fit_one_ensemble(self,X):
        C=X@X.t()
        #A parameter to account for how many top neurons we sample from. It starts from 1,
        #because we choose the top neuron when possible, e.g. when we can find an ensemble
        #that is larger than min ensemble size. If there is no ensemble with the top neuron
        #we increase the number of neurons to sample from.
        self.n_neurons_for_sampling=1
        top_neurons=self.sorting_for_seed(C)

    def sorting_for_seed(self,C):
        '''
        This function sorts the similarity matrix C to find neurons that are most correlated
        to their nr_neurons_to_av neighbors (we average over the neighbors).
        '''
        nr_neurons_to_av=self.options_dict['seed_neuron_av_nr']
        sorted_similarities,_=C.sort(dim=1)
        sorted_similarities=sorted_similarities[:,:-1][:,self.sz[0]-nr_neurons_to_av-1:]
        print(sorted_similarities.size())
        average_similarities=sorted_similarities.mean(dim=1)
        top_neurons=average_similarities.argsort()
        print('top neurons', top_neurons)
        return top_neurons

    def fit_transform(self,X):
        '''
        X-- shape (neurons, timepoints)
        '''
        X=torch.cuda.FloatTensor(X)
        X=self.zscore(X)
        #print(X)
        self.sz=X.size()
        self.fit_one_ensemble(X)
        

 
