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

    def calculate_cost_delta(self,C_summed,current_v):
        cost_delta=torch.clamp(C_summed,min=0,max=None)**2/(self.sz[1]*((current_v**2).sum()))-self.lambd
        return cost_delta

    def mask_cost_delta(self,selected_neurons,cost_delta):
        mask=torch.zeros([selected_neurons.size()[0]]).type(torch.cuda.FloatTensor)
        mask[selected_neurons==0]=1
        mask[selected_neurons!=0]=0
        masked_cost_delta=mask*cost_delta
        return masked_cost_delta

    def sum_C(self,C,selected_neurons,n):
        C_summed=(1./n)*torch.sum(C[:,selected_neurons],dim=1)
        return C_summed

    def fit_one_ensemble(self,X):
        C=X@X.t()
        #A parameter to account for how many top neurons we sample from. It starts from 1,
        #because we choose the top neuron when possible, e.g. when we can find an ensemble
        #that is larger than min ensemble size. If there is no ensemble with the top neuron
        #we increase the number of neurons to sample from.
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
            selected_neurons=torch.zeros([self.sz[0]]).type(torch.ByteTensor)
            selected_neurons[seed]=1
            #Seed current_v
            current_v=X[seed,:].flatten()
            #Fake cost to initiate while loop
            max_cost_delta=1000
            while max_cost_delta>0:
                C_summed=self.sum_C(C,selected_neurons,n)
                cost_delta=self.calculate_cost_delta(C_summed,current_v)
                #invert the 0's and 1's in the array which stores which neurons have already 
                #been selected into the assembly to use it as a mask
                masked_cost_delta=self.mask_cost_delta(selected_neurons,cost_delta)
                max_delta_neuron=masked_cost_delta.argmax()
                max_cost_delta=masked_cost_delta.max()
                if max_delta_cost>0:
                    selected_neurons[max_delta_neuron.item()]=1
                    current_v=X[(selected_neurons == 1),:].mean(dim=0)
                    n+=1
            print('pytorch current v', current_v)
            n=100000000

    def sample_seed_neuron(self,top_neurons):
        idx=torch.randint(0,self.n_neurons_for_sampling,size=(1,))
        top_neurons=top_neurons[self.sz[0]-(self.n_neurons_for_sampling):]
        seed=top_neurons[idx[0]].item()
        return seed

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
        self.sz=X.size()
        self.fit_one_ensemble(X)
        

 
