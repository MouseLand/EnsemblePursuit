import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF as NMF_sklearn
import torch.optim as optim
from sklearn.decomposition import PCA

class NMF(nn.Module):
    def __init__(self, nr_neurons, nr_timepoints, nr_components,initialization,X):
        super(NMF, self).__init__()
        if initialization=='random':
            #zeros_for_U=np.random.choice([0,1], nr_neurons*nr_components, p=[1-0.01, 0.01]).reshape((nr_neurons,nr_components))
            self.U = nn.Parameter(0.1*torch.randn(nr_neurons, nr_components, requires_grad=True))
            self.V = nn.Parameter(torch.randn(nr_components, nr_timepoints, requires_grad=True))
        if initialization=='NMF':
            X[X<0]=0
            X=X.T
            model = NMF_sklearn(n_components=nr_components, init='nndsvd', random_state=7)
            self.V=nn.Parameter(torch.tensor(model.fit_transform(X).T,requires_grad=True,dtype=torch.float32))
            print(self.V.size())
            self.U=nn.Parameter(torch.tensor(model.components_.T,requires_grad=True,dtype=torch.float32))
            print(self.U.size())
        if initialization=='PCA':
            X=X.T
            model=PCA(n_components=nr_components)
            self.V=nn.Parameter(torch.tensor(model.fit_transform(X).T,requires_grad=True,dtype=torch.float32))
            print(self.V.size())
            self.U=nn.Parameter(torch.tensor(model.components_.T,requires_grad=True,dtype=torch.float32))
            print(self.U.size())
    def forward(self):
        return self.U@self.V

def fit_NMF(X,nr_components,n_epoch):
    nr_timepoints=X.shape[1]
    nr_neurons=X.shape[0]
    X_torch=torch.tensor(X,dtype=torch.float32)
    Y=X_torch
    nmf = NMF(nr_neurons, nr_timepoints, nr_components,initialization='PCA',X=X)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_lst=[]
    optimizer = optim.SGD(nmf.parameters(), lr=0.1,momentum=0.9)
    for epoch in range(n_epoch):
        #Y_ = nmf()
        Y_ = nmf()
        loss = loss_fn(Y_, Y)
        optimizer.zero_grad() # need to clear the old gradients
        loss.backward()
        loss_lst.append(loss.item())
        optimizer.step()
        '''
        for param in nmf.parameters():
            param.data = param.data - 0.1* param.grad
            if param is nmf.U:
                param.data = torch.clamp(param.data,min=0)
        '''
        print('Epoch number:', epoch)
        print('Current loss:',loss.item())
    plt.plot(loss_lst)
    return np.array(nmf.U.data),np.array(nmf.V.data)
