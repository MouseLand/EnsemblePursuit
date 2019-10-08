import torch
import numpy as np
from scipy.sparse.linalg import eigsh

def zscore(X, axis=0):
    mean_X= np.mean(X,axis=axis)
    std_X = np.std(X, axis=axis) + 1e-10
    X -= np.expand_dims(mean_X, axis)
    X /= np.expand_dims(std_X, axis)

    return X

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


def stimulus_correlation(V, istim):
    x_train,x_test,y_train,y_test= test_train_split(V,istim)
    cc = np.mean(zscore(x_train) * zscore(x_test), axis=0)
    return cc

def evaluate_model(V,istim):
    x_train,x_test,y_train,y_test= test_train_split(V,istim)

    corr_mat = corr_matrix(x_train.T, x_test.T)
    print(np.mean(np.argmax(corr_mat, axis=0) == np.arange(0,x_train.shape[0],1,int)))


def corr_matrix(x,y):
    '''
    calculate correlation matrix
    '''

    x = x- np.mean(x,axis=0)
    y = y- np.mean(y,axis=0)

    x /= np.std(x,axis=0) + 1e-10
    y /= np.std(y,axis=0) + 1e-10

    c = x.T @ y

    return c


def subtract_spont(spont,resp):
    #print(spont)
    mu = spont.mean(axis=0)
    sd = spont.std(axis=0) + 1e-6
    resp = (resp - mu) / sd
    spont = (spont - mu) / sd
    sv,u = eigsh(spont.T @ spont, k=32)
    resp = resp - (resp @ u) @ u.T
    return resp
