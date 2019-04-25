import torch
import numpy as np
from scipy.sparse.linalg import eigsh

def test_train_split(data,stim):
    unique, counts = np.unique(stim.flatten(), return_counts=True)
    count_dict=dict(zip(unique, counts))

    keys_with_enough_data=[]
    for key in count_dict.keys():
        if count_dict[key]==2:
            keys_with_enough_data.append(key)

    filtered_stims=np.isin(stim.flatten(),keys_with_enough_data)

    #Arrange data so that responses with the same stimulus are adjacent
    z=stim.flatten()[np.where(filtered_stims)[0]]
    sortd=np.argsort(z)
    istim=np.sort(z)
    X=data[filtered_stims,:]
    out=X[sortd,:].copy()

    x_train=out[::2,:]
    y_train=istim[::2]
    x_test=out[1::2,:]
    y_test=istim[1::2]
    
    return x_train, x_test, y_train, y_test

def PCA(images,k=100):
    images=torch.cuda.FloatTensor(images)
    mean_im=torch.mean(images,dim=0)
    centered=torch.sub(images,mean_im)
    print(centered.size())
    U,S,V=torch.svd(centered)
    #print(U,S,V)
    S=torch.diag(S)
    print(U.size())
    reduced=torch.matmul(U[:,:k],S[:k,:k])
    #print(reduced.size())
    reduced=torch.matmul(reduced,V[:,:k].t())
    return np.array(reduced.cpu())

def evaluate_model(x_train,x_test):
    corr_mat=np.zeros((x_train.shape[0],x_train.shape[0]))
    for j in range(0,x_train.shape[0]):
        for i in range(0,x_test.shape[0]):
            corr_mat[j,i]=np.corrcoef(x_train[j,:],x_test[i,:])[0,1]
    print(np.mean(np.argmax(corr_mat, axis=0) == np.arange(0,x_train.shape[0],1,int)))
    

def corrcoef(x,y):
    '''
    Torch implementation of the full correlation matrix.
    '''
    # calculate covariance matrix of columns
    mean_x = torch.mean(x,0)
    xm = torch.sub(x,mean_x)
    mean_y=torch.mean(y,0)
    ym=torch.sub(y,mean_y)
    c = torch.matmul(x.t(),y)
    c = c / (x.size(0))

    # normalize covariance matrix
    std_x=torch.std(x,0)
    std_y=torch.std(y,0)
    std=torch.matmul(std_x.view(std_x.size()[0],1),std_y.view(1,std_y.size()[0]))
    c = c.div(std)
    return c

def evaluate_model_torch(x_train,x_test):
    x_train=torch.cuda.FloatTensor(x_train).t()
    x_test=torch.cuda.FloatTensor(x_test).t()
    corr_mat=np.array(corrcoef(x_train,x_test).cpu())
    #print(corr_mat.size())
    x_train=np.array(x_train.t().cpu())
    print(corr_mat.shape)
    return np.mean(np.argmax(corr_mat, axis=0) == np.arange(0,x_train.shape[0],1,int))

def subtract_spont(spont,resp):
    #print(spont)
    mu = spont.mean(axis=0)
    sd = spont.std(axis=0) + 1e-6
    resp = (resp - mu) / sd
    spont = (spont - mu) / sd
    sv,u = eigsh(spont.T @ spont, k=32)
    resp = resp - (resp @ u) @ u.T
    return resp

   
