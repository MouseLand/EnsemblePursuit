import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import skew
import time
from scipy.stats import zscore

def new_ensemble(X, C, seed_timecourse, lam, discard_first_neuron = False):
    # X are the NT by NN neural activity traces (z-scored)
    # C is the covariance matrix of X
    # seed_timecourse initializes the ensemble
    # lam is the explained variance threshold
    # discard the first neuron in the ensemble (if this was used to seed the pursuit)

    NT, NN = X.shape
    mask_neurons=np.ones((NN,),dtype=bool)

    # compute initial bias
    bias = seed_timecourse @ X
    current_v = seed_timecourse

    # initialize C_summed
    C_summed = bias.flatten()

    # keep track of neuron order
    iorder = np.zeros(NN, 'int32')

    n = 0
    while True:
        # at each iteration, first determine the neuron to be added
        imax = np.argmax(C_summed * mask_neurons)

        # compute norm of ensemble trace
        vnorm = np.sum(current_v**2)

        # compute delta cost function
        cost_delta = np.maximum(0., C_summed[imax])**2 / vnorm

        # if cost/variance explained is less than lam (* n_timepoints) then break
        if cost_delta<lam*NT:
            break

        # zero out freshly added neuron
        mask_neurons[imax] = False

        if n==0 and discard_first_neuron:
            discard_first_neuron = False
            continue

        # add column of C
        C_summed = C_summed + C[:, imax]

        # add column of X
        current_v = current_v + X[:, imax]

        # keep track of neurons in ensembles
        iorder[n] = imax

        n = n+1

    # take only first n neurons
    iorder = iorder[:n]

    return iorder, current_v

def one_round_of_kmeans(V, X, lam=0.01, threshold=True):
    # V are the NT by nK cluster activity traces
    # X are the NT by NN neural activity traces (z-scored)
    # if the threshold is true, neurons only make it into a cluster if their explained variance is above lam
    # this is useful in the last stages when most neurons only have noise left

    NT, nK = V.shape

    # computes projections of neurons onto components
    cc = V.T @ X

    # take the biggest projection component for each neuron
    imax = np.argmax(cc, axis=0)

    # for every neuron, compute max projection
    w = np.max(cc, axis=0)

    # explained variance for each neuron
    amax = np.maximum(0, w)**2/NT

    # initialize total explained variance for each cluster
    vm = np.zeros((nK,))

    # update each cluster in k-means
    for j in range(nK):
        # take all neurons assigned to this cluster
        ix = imax==j
        if threshold:
            ix = np.logical_and(ix, amax>lam)

        # if there are more than 0 neurons assigned
        if np.sum(ix)>0:
            # update the component to be the mean  of assigned neurons
            V[:,j] = X[:, ix] @ w[ix]

            # compute total explained variance for this cluster
            vm[j] = np.sum(amax[ix])

    # re-normalize each column of V separately
    V = V/np.sum(V**2 + 1e-6, axis=0)**.5

    return V, vm

def one_round_of_PCA(V, C):
    # computes projections of neurons onto components
    for t in range(5):
        V = C @ V
        V /= np.sum(V**2)**.5
        V = V.flatten()

    return V

def initialize_kmeans(X, nK, lam):
    # initialize k-means for matrix X (NT by NN)
    # the columns of X should be Z-SCORED

    # initialize k-means centers. THINK HOW TO MAKE THIS DETERMINISTIC
    t0 = time.time()
    model = PCA(n_components=nK, random_state = 101).fit(X.T)
    V = model.components_.T
    V = V * np.sign(skew(V.T @ X,axis=1))
    print('obtained %d PCs in %2.4f seconds'%(nK, time.time()-t0))

    #np.random.seed(101)
    #rperm = np.random.permutation(X.shape[1])
    #V = X[:, rperm[:nK]]
    #V = np.random.randn(X.shape[0], nK)

    # keep V as unit norm vectors
    V /= np.sum(V**2 + 1e-6, axis=0)**.5

    t0 = time.time()

    # 10 iterations is plenty
    for j in range(10):
        # run one round of k-means and update the cluster activities (V)
        V, vm = one_round_of_kmeans(V, X, lam, j>5)

    print('initialized %d clusters with k-means in %2.4f seconds'%(nK, time.time()-t0))

    return V, vm

class EnsemblePursuit():
    def __init__(self,n_components,lam):
        self.n_components=n_components
        self.lam=lam

    def fit(self,X, nKmeans=25):
        X = (X - np.mean(X,axis=0)) / (1e-5 + np.std(X, axis=0))
        nK=self.n_components
        lam=self.lam

        NT, NN = X.shape

        # convert to float64 for numerical precision
        X = np.float64(X)

        # initialize k-means clusters and compute their variance in vm
        V, vm = initialize_kmeans(X, nKmeans, lam)

        # initialize vectors in ensemble pursuit (Vs)
        vs = np.zeros((NT, nK))

        # initialize U
        U = np.zeros((NN, nK))

        # precompute covariance matrix of neurons
        C = X.T @ X

        # keep track of number of neurons per ensemble
        ns = np.zeros(nK,)

        # time the ensemble pursuit
        t0 = time.time()

        #  outer loop
        for j in range(nK):
            # initialize with "biggest" k-means ensemble (by variance)
            imax = np.argmax(vm)

            # zscore the seed trace
            seed = zscore(V[:, imax])

            # fit one ensemble starting from this seed
            iorder, current_v  = new_ensemble(X, C, seed, lam)

            # keep track of number of neurons
            ns[j] = len(iorder)

            # normalize current_v to unit norm
            current_v /= np.sum(current_v**2)**.5

            # update column of Vs
            vs[:,j] = current_v

            # projection of each neuron onto this ensemble trace
            w = current_v @ X

            # update weights for neurons in this ensemble
            U[iorder, j] = w[iorder]

            # update activity trace
            X[:, iorder] -= np.outer(current_v, w[iorder])

            # rank one update to C
            wtw = np.outer(w[iorder],  w)

            # update the columns
            C[:, iorder] -= wtw.T

            # update the rows
            C[iorder, :] -= wtw

            # add back term for the submatrix of neurons in this ensemble
            C[iorder[:, np.newaxis], iorder] += wtw[:, iorder]

            # run one round of k-means because we changed X
            V, vm = one_round_of_kmeans(V, X, lam)

            if j%25==0 or j == nK-1:
                print('ensemble %d, time %2.2f, nr neurons %d, EV %2.4f'%(j, time.time() - t0, len(iorder), 1-np.mean(X**2)))
        print('average sparsity is %2.4f'%(np.mean(U>1e-5)))

        return U, vs
