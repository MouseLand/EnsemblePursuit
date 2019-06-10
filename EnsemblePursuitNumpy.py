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
   
    def fit_transform(self,X):
        X=self.zscore(X)
        print(X)
