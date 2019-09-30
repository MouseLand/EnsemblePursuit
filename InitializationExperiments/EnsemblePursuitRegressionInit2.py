import torch
import numpy as np

class EnsemblePursuitRegressionInit():
    def __init__(self, n_ensembles, lambd, options_dict):
        self.n_ensembles=n_ensembles
        self.lambd=lambd
        self.options_dict=options_dict

    def zscore(self,X):
        #Have to transpose X to make torch.sub and div work. Transpose back into 
        #original shape when done with calculations. 
        mean_stimuli=X.t().mean(dim=0)
        std_stimuli=X.t().std(dim=0)+1e-10
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

    def sum_C(self,C_summed_unnorm,C,max_delta_neuron):
        C_summed_unnorm=C_summed_unnorm+C[:,max_delta_neuron]
        return C_summed_unnorm

    def sum_v(self, v, max_delta_neuron, X):
        current_v=v+X[max_delta_neuron,:]
        return current_v
    
    def linear_regression_torch(self,X,y):
        X=X.cuda().t()
        y=y.t()
        weights=torch.pinverse(X)@y
        res=y-X@weights
        #print(res.size())
        norm=torch.norm(res,p=2,dim=0)
        #print(norm.size())
        residuals,neurons=norm.sort()
        return neurons

    def fit_one_ensemble(self,X):
        C=X@X.t()
        #A parameter to account for how many top neurons we sample from. It starts from 1,
        #because we choose the top neuron when possible, e.g. when we can find an ensemble
        #that is larger than min ensemble size. If there is no ensemble with the top neuron
        #we increase the number of neurons to sample from.
        self.n_neurons_for_sampling=1
        top_neurons=self.linear_regression_torch(self.V,X)
        n=0
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
            current_v_unnorm=current_v.clone()
            #Fake cost to initiate while loop
            max_cost_delta=1000
            C_summed_unnorm=0
            max_delta_neuron=seed
            while max_cost_delta>0:
                #Add the x corresponding to the max delta neuron to C_sum. Saves computational 
                #time.
                C_summed_unnorm=self.sum_C(C_summed_unnorm,C,max_delta_neuron)
                C_summed=(1./n)*C_summed_unnorm
                cost_delta=self.calculate_cost_delta(C_summed,current_v)
                #invert the 0's and 1's in the array which stores which neurons have already 
                #been selected into the assembly to use it as a mask
                masked_cost_delta=self.mask_cost_delta(selected_neurons,cost_delta)
                max_delta_neuron=masked_cost_delta.argmax()
                max_cost_delta=masked_cost_delta.max()
                if max_delta_cost>0:
                    selected_neurons[max_delta_neuron]=1
                    current_v_unnorm= self.sum_v(current_v_unnorm,max_delta_neuron,X)
                    n+=1
                    current_v=(1./n)*current_v_unnorm
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
        current_u=torch.zeros((X.size(0),1))
        current_u[selected_neurons,0]=torch.clamp(C_summed[selected_neurons].cpu(),min=0,max=None)/(current_v**2).sum()
        current_u=current_u.cpu()
        current_v=current_v.cpu()
        self.U=torch.cat((self.U,current_u.view(X.size(0),1)),1)
        self.V=torch.cat((self.V,current_v.view(1,X.size(1))),0)
        print(n)
        return current_u, current_v
            

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
        average_similarities=sorted_similarities.mean(dim=1)
        top_neurons=average_similarities.argsort()
        return top_neurons

    def fit_transform(self,X):
        '''
        X-- shape (neurons, timepoints)
        '''
        X=torch.cuda.FloatTensor(X)
        X=self.zscore(X)
        self.sz=X.size()
        #Initializes U and V with zeros, later these will be discarded.
        self.U=torch.zeros((X.size(0),1))
        self.V=torch.zeros((1,X.size(1)))
        for iteration in range(0,self.n_ensembles):
            current_u, current_v=self.fit_one_ensemble(X)
            U_V=current_u.reshape(self.sz[0],1)@current_v.reshape(1,self.sz[1])
            X=X.cpu()-U_V
            X=X.cuda()
            print('ensemble nr', iteration)
            cost=torch.mean(torch.mul(X,X))
            print('cost',cost)
        #After fitting arrays discard the zero initialization rows and columns from U and V.
        self.U=self.U[:,1:]
        self.V=self.V[1:,:] 
        return self.U,self.V.t()
        

 
