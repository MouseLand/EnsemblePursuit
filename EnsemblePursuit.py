import torch
import numpy as np

class EnsemblePursuitPyTorch():
    
    def zscore(self):
        mean_stimuli=self.X.mean(dim=0)
        std_stimuli=self.X.std(dim=0)+0.0000000001
        
        self.X=torch.sub(self.X,mean_stimuli)
        self.X=self.X.div(std_stimuli)
    
    def calculate_cost_delta(self):
        cost_delta=(torch.clamp(torch.matmul(self.current_v,self.X),min=0,max=None)**2)/(self.sz[0]*torch.matmul(self.current_v,self.current_v))-self.lambd
        #print('cost delta',cost_delta.mean())
        return cost_delta
    
    def fit_one_assembly(self):
        '''
        Function for fitting one cell assembly and computing u and v of the currrent assembly (self.current_u,
        self.current_v).
        One neuron cell assemblies are excluded. 
        '''
        with torch.cuda.device(0) as device:
            #Fake i for initiating while loop. self.i stores the number of neurons in assemblies.
            self.i=7
            #If i is 1, e.g. only one neuron in fit cell assembly, will run fitting the assembly again. 
            #safety it to avoid infinite loops.
            safety_it=0
            n_of_neurons=self.neuron_init_dict['parameters']['n_of_neurons']
            min_assembly_size=self.neuron_init_dict['parameters']['min_assembly_size']
            #Correlate correlation matrix if top_k_corr method of assembly initialization, because
            #then sampling will be better
            if self.neuron_init_dict['method']=='top_k_corr':
                self.for_sampling,self.vals=self.corr_top_k(n_neurons=n_of_neurons)
            #Reject assemblies with less than 8 neurons
            max_delta_cost=1000
            while max_delta_cost>0:
                self.ep_lst=[]
                if self.first_assembly==True:
                    top_neurons=self.select_top_neurons()
                elif self.first_assembly==False and self.neuron_init_dict['method']=='from_time_point':
                    self.neuron_init_dict['method']='top_k_corr'
                    top_neurons=self.select_top_neurons()
                elif self.first_assembly==False:
                    top_neurons=self.select_top_neurons()
                #Array of keeping track of neurons in the cell assembly
                self.selected_neurons=torch.zeros([self.sz[1]]).cuda()
                print('Top neurons', top_neurons)
                for j in range(0,len(top_neurons)):
                    self.selected_neurons[top_neurons[j]]=1
                #Seed current_v
                self.current_v=self.X[:,top_neurons].flatten()#.mean(1).flatten()
                #Fake cost to initiate while loop
                max_delta_cost=1000
                #reset i
                self.i=1
                self.neuron_lst=[top_neurons[0]]
                while max_delta_cost>0:
                    cost_delta=self.calculate_cost_delta()
                    #invert the 0's and 1's in the array which stores which neurons have already 
                    #been selected into the assembly to use it as a mask
                    mask=self.selected_neurons.clone()
                    mask[self.selected_neurons==0]=1
                    mask[self.selected_neurons!=0]=0
                    masked_cost_delta=mask*cost_delta
                    values,sorted_neurons=masked_cost_delta.sort()
                    max_delta_neuron=sorted_neurons[-1]
                    self.ep_lst.append(max_delta_neuron.item())
                    max_delta_cost=values[-1]
                    if max_delta_cost>0:
                        self.selected_neurons[max_delta_neuron.item()]=1
                        self.current_v= self.X[:, (self.selected_neurons == 1)].mean(dim=1)
                        self.neuron_lst.append(max_delta_neuron.item())
                        #print('sel neurons', self.X[:, (self.selected_neurons == 1)].size())
                        self.i+=1
                safety_it+=1
                #Increase number of neurons to sample from if while loop hasn't been finding any assemblies.
                '''
                if safety_it>100:
                    self.neuron_init_dict['parameters']['n_of_neurons']=500
                    if self.neuron_init_dict['method']=='top_k_corr':
                        self.for_sampling,self.vals=self.corr_top_k(n_neurons=500)
                if safety_it>600:
                    self.neuron_init_dict['parameters']['n_of_neurons']=1000
                    if self.neuron_init_dict['method']=='top_k_corr':
                        self.for_sampling,self.vals=self.corr_top_k(n_neurons=1000)
                if safety_it>1600:
                    raise ValueError('Assembly capacity too big, can\'t fit model')
                '''
            #Once one assembly has been found, set the variable to false
            self.first_assembly=False
            #Add final seed neuron to seed_neurons.        
            self.seed_neurons=self.seed_neurons+top_neurons          
            #Calculate u based on final v fit for a cell assembly. 
            #torch.set_printoptions(threshold=5000)
            #self.current_u=torch.cuda.FloatTensor(self.sz[1]).fill_(0)
            ind=torch.zeros(self.sz[1]).type(torch.ByteTensor)
            ind[self.neuron_lst]=1
            #X_for_U=self.X[:,ind]
            #u_mul=torch.clamp(torch.matmul(self.current_v,X_for_U),min=0,max=None)/torch.matmul(self.current_v,self.current_v)
            self.current_u=torch.clamp((1./self.i)*torch.sum(self.c[:,(self.selected_neurons==1)],dim=1),min=0,max=None)/torch.matmul(self.current_v,self.current_v)
            #self.current_u[ind]=u_mul
            print(self.current_u)
            self.ensemble_neuron_lst.append(self.neuron_lst)
            self.U=torch.cat((self.U,self.current_u.view(self.X.size(1),1)),1)
            self.V=torch.cat((self.V,self.current_v.view(1,self.X.size(0))),0)
            
    def select_top_neurons(self):
        if self.neuron_init_dict['method']=='top_k_corr':
            n_of_neurons=self.neuron_init_dict['parameters']['n_of_neurons']
            top_neurons=self.select_top_k_corr_neuron(self.for_sampling,self.vals,n_of_neurons)
        if self.neuron_init_dict['method']=='random':
            top_neurons=[np.random.randint(0,self.sz[1],1)[0]]
        if self.neuron_init_dict['method']=='from_time_point':
            top_neurons=self.select_from_time_point()
        #For the suite2p gui if seed neurons are provided
        if self.neuron_init_dict['method']=='gui_selected':
            top_neurons=self.gui_selected_neurons
        return top_neurons
    
    def select_from_time_point(self):
        threshold=self.neuron_init_dict['parameters']['T']
        threshold_array=(self.original_X>=threshold).sum(dim=1)
        #print('thr_array',threshold_array)
        values,sorted_timepoints=threshold_array.sort()
        timepoint=sorted_timepoints[-1]
        #print('t',timepoint)
        neurons=(self.original_X[timepoint,:]>=threshold).nonzero()
        #print('neurons',neurons)
        return neurons.tolist()
    
    def corrcoef(self,x):
        '''
        Torch implementation of the full correlation matrix.
        '''
        # calculate covariance matrix of columns
        mean_x = torch.mean(x,0)
        xm = torch.sub(x,mean_x)
        c = x.mm(x.t())
        self.c=c.clone()
        #print(self.c.size(0))
        c = c / (x.size(1))

        # normalize covariance matrix
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())
        #print((c!=c).nonzero())
        # clamp between -1 and 1
        c = torch.clamp(c, -1.0, 1.0)

        return c
    
    def corr_top_k(self,n_neurons=100):
        '''
        Finds n_neurons neurons that are on average most correlated to their 
        5 closest neighbors.
        '''
        n_av_neurons=self.neuron_init_dict['parameters']['n_av_neurons']
        #Compute full correlation matrix (works with one neuron per column,
        #so have to transpose.)
        corr=self.corrcoef(self.X.t())
        #Sorts each row of correlation matrix
        vals,ix=corr.sort(dim=1)
        #Discards the last entry corresponding to the diagonal 1 and then
        #selects n_av_neurons of the largest entries from sorted array.
        top_vals=vals[:,:-1][:,self.sz[1]-n_av_neurons+1:]
        #Averages the 5 top correlations.
        av=torch.mean(top_vals,dim=1)
        #Sorts the averages
        vals,top_neurons=torch.sort(av)
        #Selects top neurons
        top_neuron=top_neurons[self.sz[1]-(n_neurons+1):]
        print('Neurons',self.sz[1]-(n_neurons+1),top_neuron)
        top_val=vals[self.sz[1]-(n_neurons+1):]
        return top_neuron,top_val
          
    
    def select_top_k_corr_neuron(self,top_neuron,top_val,n_neurons=100):
        '''
        Randomly samples from k top correlated urons.
        '''
        #Randomly samples a neuron from the n_of_neurons top correlated.
        idx=torch.randint(0,n_neurons,size=(1,))
        #print('top n', top_neuron[idx[0]].item(), top_val[idx[0]].item())
        return [top_neuron[idx[0]].item()]
    
    
    def fit_transform(self,X,lambd,n_ensembles,neuron_init_dict):
        torch.manual_seed(7)
        with torch.cuda.device(0) as device:
            self.ensemble_neuron_lst=[]
            self.neuron_init_dict=neuron_init_dict
            self.lambd=lambd
            #Creates cuda tensor from data
            self.X=torch.cuda.FloatTensor(X)
            #z-score data.
            self.zscore()
            #Keep original data for one type of initialization
            if self.neuron_init_dict['method']=='from_time_point':
                self.original_X=self.X.clone()
            #Store dimensionality of X for later use.
            self.sz=self.X.size()
            #print('sz',self.sz)
            #Initializes U and V with zeros, later these will be discarded.
            self.U=torch.zeros((self.X.size(1),1)).cuda()
            self.V=torch.zeros([1,self.X.size(0)]).cuda()
            #List for storing the number of neurons in each fit assembly.
            self.nr_of_neurons=[]
            #List for storing the seed neurons for each assembly.
            self.seed_neurons=[]
            cost_lst=[]
            #a variable to switch to random initialization after finding first assembly if the method is
            #selecting neurons from a time point
            self.first_assembly=True
            for iteration in range(0,n_ensembles):
                self.fit_one_assembly()
                self.nr_of_neurons.append(self.i)
                U_V=torch.mm(self.current_u.view(self.sz[1],1),self.current_v.view(1,self.sz[0]))
                U_V[U_V != U_V] = 0
                self.X=(self.X-U_V.t())
                print('ensemble nr', iteration)
                #print('u',self.current_u)
                #print('v',self.current_v)
                #print('length v', torch.matmul(self.current_v,self.current_v))
                #print('norm',torch.norm(self.X))
                self.cost=torch.mean(torch.mul(self.X,self.X))
                print('cost',self.cost)
                cost_lst.append(self.cost.item())
            #After fitting arrays discard the zero initialization rows and columns from U and V.
            self.U=self.U[:,1:]
            self.V=self.V[1:,:]
            #print(self.X.size())
            #print(self.U.size())
            #print(self.V.size())
            return torch.matmul(self.U,self.V).t().cpu(), self.nr_of_neurons, self.U.cpu(), np.array(self.V.cpu()).T, cost_lst, self.seed_neurons, self.ensemble_neuron_lst 

    def fit_transform_suite2p(self, X, lambd, neuron_init_dict,gui_selected_neurons=None):
        self.X=torch.cuda.FloatTensor(X)
        self.zscore()
        self.lambd=lambd
        self.neuron_init_dict=neuron_init_dict
        if gui_selected_neurons!=None:
            self.gui_selected_neurons=gui_selected_neurons
        if self.neuron_init_dict['method']=='from_time_point':
                self.original_X=self.X.clone()
        #Store dimensionality of X for later use.
        self.sz=self.X.size()
        #Initializes U and V with zeros, later these will be discarded.
        self.U=torch.zeros((self.X.size(1),1)).cuda()
        self.V=torch.zeros([1,self.X.size(0)]).cuda()   
        self.nr_of_neurons=[]
        #List for storing the seed neurons for each assembly.
        self.seed_neurons=[]
        cost_lst=[]
        #a variable to switch to random initialization after finding first assembly if the method is
        #selecting neurons from a time point
        self.first_assembly=True 
        self.fit_one_assembly()    
        return self.ep_lst

            

