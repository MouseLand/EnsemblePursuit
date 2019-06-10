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

    def fit_transform(self,X):
        '''
             X-- shape (neurons, timepoints)
        '''
        X=torch.cuda.FloatTensor(X)
        X=self.zscore(X)
        print(X)
        

 
