import numpy as np
import time
from sklearn.cluster import KMeans


class EnsemblePursuitNumpyFast():
    def __init__(self,n_ensembles,lambd,options_dict):
        self.n_ensembles=n_ensembles
        self.lambd=lambd
        self.options_dict=options_dict

    def zscore(self,X):
        mean_stimuli=np.mean(X.T,axis=0)
        std_stimuli=np.std(X.T,axis=0,ddof=1)+1e-10
        X=np.subtract(X.T,mean_stimuli)
        X=np.divide(X,std_stimuli)
        return X.T

    def calculate_dot_squared(self,C_summed):
        '''
        The matrix of summed correlation entries, which represent x.T@v have to be
        squared for computing the cost.
        '''
        dot_squared=np.clip(C_summed,a_min=0,a_max=None)**2
        return dot_squared

    def calculate_cost_delta(self,C_summed,current_v):
        '''
        Use the corrrelation matrix to compute the change in cost for adding one more neuron.
        '''
        cost_delta=np.clip(C_summed,a_min=0,a_max=None)**2/(self.sz[1]*(current_v**2).sum())-self.lambd
        return cost_delta

    def mask_dot_squared(self,selected_neurons,dot_squared):
        '''
        Mask out neurons that are already in the ensemble for cost computation.
        '''
        mask=np.zeros((selected_neurons.shape[0]),dtype=bool)
        mask[selected_neurons==0]=1
        mask[selected_neurons!=0]=0
        masked_dot_squared=mask*dot_squared
        return masked_dot_squared

    def sum_C(self,C_summed_unnorm,C,max_delta_neuron):
        '''
        As you accumulate more neurons to the ensemble you sum columns of the
        correlation matrix into an accumulator matrix.
        '''
        C_summed_unnorm=C_summed_unnorm+C[:,max_delta_neuron]
        return C_summed_unnorm

    def sum_v(self, v, max_delta_neuron, X):
        current_v=v+X[max_delta_neuron,:]
        return current_v

    def update_C(self,X,prev_C,u,v,selected_neurons):
        #selected_neurons=np.nonzero(u)[0]
        C=prev_C
        cross_term_init=X@v
        cross_term=np.outer(u[selected_neurons],cross_term_init)
        C[selected_neurons,:]=C[selected_neurons,:]-cross_term
        ixgrid=np.ix_(~selected_neurons,selected_neurons)
        C[ixgrid]=C[ixgrid]-cross_term.T[~selected_neurons,:]
        return C

    def fit_one_ensemble(self,X,C):
        #A parameter to account for how many top neurons we sample from. It starts from 1,
        #because we choose the top neuron when possible, e.g. when we can find an ensemble
        # that is larger than min ensemble size. If there is no ensemble with the top neuron
        # we increase the number of neurons to sample from.
        self.n_neurons_for_sampling=1
        n=0
        min_assembly_size=self.options_dict['min_assembly_size']
        #index for switching between top neurons for fitting ensemble when the first neurons
        #doesn't give large enough ensemble
        index=-1
        #A while loop for trying sampling other neurons if the found ensemble size is smaller
        #than threshold.
        while n<min_assembly_size:
            seed=self.repeated_seed(C,index)
            n=1
            current_v=X[seed,:]
            current_v_unnorm=current_v.copy()
            selected_neurons=np.zeros((X.shape[0]),dtype=bool)
            #Seed current_v
            selected_neurons[seed]=1
            #Fake cost to initiate while loop
            max_cost_delta=1000
            C_summed_unnorm=0
            max_delta_neuron=seed
            while max_cost_delta>0:
                #Add the x corresponding to the max delta neuron to C_sum. Saves computational
                #time.
                print(n)
                C_summed_unnorm=self.sum_C(C_summed_unnorm,C,max_delta_neuron)
                C_summed=(1./n)*C_summed_unnorm
                dot_squared=self.calculate_dot_squared(C_summed)
                #invert the 0's and 1's in the array which stores which neurons have already
                #been selected into the assembly to use it as a mask
                masked_dot_squared=self.mask_dot_squared(selected_neurons,dot_squared)
                max_delta_neuron=np.argmax(masked_dot_squared)
                cost_delta=self.calculate_cost_delta(C_summed[max_delta_neuron],current_v)
                if cost_delta>0:
                    selected_neurons[max_delta_neuron]=1
                    current_v_unnorm= self.sum_v(current_v_unnorm,max_delta_neuron,X)
                    n+=1
                    current_v=(1./n)*current_v_unnorm
                max_cost_delta=cost_delta

            index+=-1
        print('nr of neurons in ensemble',n)
        current_u=np.zeros((X.shape[0],1))
        current_u[selected_neurons,0]=np.clip(C_summed[selected_neurons],a_min=0,a_max=None)/(current_v**2).sum()
        self.U=np.concatenate((self.U,current_u),axis=1)
        self.V=np.concatenate((self.V,current_v.reshape(1,self.sz[1])),axis=0)
        return current_u, current_v, C, selected_neurons

    def fit_one_ensemble_suite2p(self,X,starting_v):
        X=self.zscore(X)
        C=X@X.T
        self.sz=X.shape
        #cost_vector=np.clip(X@starting_v,a_min=0,a_max=None)**2
        corr_vector=self.vcorrcoef(X,starting_v)
        seed=np.argmax(corr_vector)
        current_v=starting_v
        current_v_unnorm=current_v.copy()
        selected_neurons=np.zeros((X.shape[0]),dtype=bool)
        #Seed current_v
        selected_neurons[seed]=1
        #Fake cost to initiate while loop
        max_cost_delta=1000
        C_summed_unnorm=0
        max_delta_neuron=seed
        n=1
        while max_cost_delta>0:
            #Add the x corresponding to the max delta neuron to C_sum. Saves computational
            #time.
            print('n',n)
            C_summed_unnorm=self.sum_C(C_summed_unnorm,C,max_delta_neuron)
            C_summed=(1./n)*C_summed_unnorm
            dot_squared=self.calculate_dot_squared(C_summed)
            #invert the 0's and 1's in the array which stores which neurons have already
            #been selected into the assembly to use it as a mask
            masked_dot_squared=self.mask_dot_squared(selected_neurons,dot_squared)
            max_delta_neuron=np.argmax(masked_dot_squared)
            cost_delta=self.calculate_cost_delta(C_summed[max_delta_neuron],current_v)
            print('cost delta',cost_delta)
            if cost_delta>0:
                selected_neurons[max_delta_neuron]=1
                current_v_unnorm= self.sum_v(current_v_unnorm,max_delta_neuron,X)
                n+=1
                current_v=(1./n)*current_v_unnorm
            max_cost_delta=cost_delta
        return selected_neurons

    def fit_one_ensemble_seed_timecourse(self,X,C,seed_timecourse):
        self.n_neurons_for_sampling=1
        n=0
        min_assembly_size=self.options_dict['min_assembly_size']
        min_assembly_size=1
        #index for switching between top neurons for fitting ensemble when the first neurons
        #doesn't give large enough ensemble
        index=-1
        current_v=seed_timecourse
        current_v_unnorm=current_v.copy()
        selected_neurons=np.zeros((X.shape[0]),dtype=bool)
        #Fake cost to initiate while loop
        max_cost_delta=1000
        C_summed_unnorm=X@seed_timecourse
        n=1
        while n<min_assembly_size:
            #C_summed_unnorm=X@seed_timecourse
            #seed=self.repeated_seed(C,index)
            n=1
            selected_neurons=np.zeros((X.shape[0]),dtype=bool)
            dot_squared=self.calculate_dot_squared(C_summed_unnorm)
            max_delta_neuron=np.argmax(masked_dot_squared)
            #Seed current_v
            selected_neurons[max_delta_neuron]=1
            #Fake cost to initiate while loop
            max_cost_delta=1000
            #C_summed_unnorm=X@seed_timecourse
            while max_cost_delta>0:
                #Add the x corresponding to the max delta neuron to C_sum. Saves computational
                #time.
                print(n)
                C_summed_unnorm=self.sum_C(C_summed_unnorm,C,max_delta_neuron)
                C_summed=(1./n)*C_summed_unnorm
                dot_squared=self.calculate_dot_squared(C_summed)
                #invert the 0's and 1's in the array which stores which neurons have already
                #been selected into the assembly to use it as a mask
                masked_dot_squared=self.mask_dot_squared(selected_neurons,dot_squared)
                max_delta_neuron=np.argmax(masked_dot_squared)
                cost_delta=self.calculate_cost_delta(C_summed[max_delta_neuron],current_v)
                if cost_delta>0:
                    selected_neurons[max_delta_neuron]=1
                    current_v_unnorm= self.sum_v(current_v_unnorm,max_delta_neuron,X)
                    n+=1
                    current_v=(1./n)*current_v_unnorm
                max_cost_delta=cost_delta

            index+=-1
        print('nr of neurons in ensemble',n)
        current_u=np.zeros((X.shape[0],1))
        current_u[selected_neurons,0]=np.clip(C_summed[selected_neurons],a_min=0,a_max=None)/(current_v**2).sum()
        self.U=np.concatenate((self.U,current_u),axis=1)
        self.V=np.concatenate((self.V,current_v.reshape(1,self.sz[1])),axis=0)
        return current_u, current_v, C, selected_neurons


    def repeated_seed(self,C,index):
        nr_neurons_to_av=self.options_dict['seed_neuron_av_nr']
        sorted_similarities=np.sort(C,axis=1)[:,:-1][:,C.shape[0]-nr_neurons_to_av-1:]
        average_similarities=np.mean(sorted_similarities,axis=1)
        top_neurons=np.argsort(average_similarities)
        seed=top_neurons[-index]
        return seed

    def fit_transform(self,X):
        X=self.zscore(X)
        self.sz=X.shape
        self.U=np.zeros((X.shape[0],1))
        self.V=np.zeros((1,X.shape[1]))
        start=time.time()
        C=X@X.T
        end=time.time()
        print('full',end-start)
        for iteration in range(0,self.n_ensembles):
            start=time.time()
            current_u, current_v, C,selected_neurons=self.fit_one_ensemble(X,C)
            end=time.time()
            print(end-start,'loop')
            U_V=current_u.reshape(self.sz[0],1)@current_v.reshape(1,self.sz[1])
            start=time.time()
            C=self.update_C(X,C,current_u,current_v,selected_neurons)
            end=time.time()
            print('optimized',end-start)
            X=X-U_V
            print('ensemble nr', iteration)
            cost=np.mean(X*X)
            print('cost',cost)
        #After fitting arrays discard the zero initialization rows and columns from U and V.
        self.U=self.U[:,1:]
        self.V=self.V[1:,:]
        return self.U, self.V.T

    def fit_transform_kmeans_init(self,X):
        X=self.zscore(X)
        self.sz=X.shape
        self.U=np.zeros((X.shape[0],1))
        self.V=np.zeros((1,X.shape[1]))
        start=time.time()
        C=X@X.T
        end=time.time()
        print('full',end-start)
        start=time.time()
        #kmeans = KMeans(n_clusters=self.n_ensembles, random_state=0).fit(X.T).labels_
        #np.save('kmeans_init.npy',kmeans)
        kmeans=np.load('kmeans_init.npy')
        end=time.time()
        print('kmeans',end-start)
        for iteration in range(0,self.n_ensembles):
            start=time.time()
            average_timecourse=np.mean(X[np.where(kmeans==iteration),:], axis=0).T
            print('shp',average_timecourse.shape)
            current_u, current_v, C,selected_neurons=self.fit_one_ensemble_seed_timecourse(X,C,average_timecourse)
            end=time.time()
            print(end-start,'loop')
            U_V=current_u.reshape(self.sz[0],1)@current_v.reshape(1,self.sz[1])
            start=time.time()
            C=self.update_C(X,C,current_u,current_v,selected_neurons)
            end=time.time()
            print('optimized',end-start)
            X=X-U_V
            print('ensemble nr', iteration)
            cost=np.mean(X*X)
            print('cost',cost)
        #After fitting arrays discard the zero initialization rows and columns from U and V.
        self.U=self.U[:,1:]
        self.V=self.V[1:,:]
        return self.U, self.V.T
