
from det import DET
import numpy as np
import numpy.random as rng
from relu import ReluNet


def sample_from_prior(probcl, meancl, stdcl, nsamples, seed):
    '''Samples 'nsamples' points from the mixture of Gaussians: 

    (probcl, meancl, stdcl)

    '''
    assert(len(probcl) == len(meancl))
    assert(len(probcl) == len(stdcl))
    rng.seed(seed)
    samples = []
    for cl in rng.choice(len(probcl), nsamples, p=probcl):
        samples.append(rng.normal(meancl[cl], stdcl[cl]))

    return np.array(samples)



def generate_constraints(nx, ny, seed):
    rng.seed(seed)
    constraints = []
    indices = rng.choice(ny, nx)
    for i in range(ny):
        lincomb = []
        for j in range(nx):
            if indices[j] == i:
                k = rng.uniform()
                lincomb.append((xj, k))

        constraints.append(lincomb)

    return constraints
                
                

if __name__ == '__main__':

    seed = 666

    # problem
    xdim = 3
    ydim = 2
    ncl = 2

    # relunet
    hiddendim = 32
    reludim = (xdim, hiddendim, ydim)
    nn_train = 1000
    nn_test = 100
    nepochs = 100
    threshold = 0.0

    # prior model
    nmin = 10
    nmax = 50
    det_train = 5000
    det_valid = 500
    

    rng.seed(seed)
    # Gaussian mixture as a prior
    wcl = rng.uniform(size=ncl)
    probcl = wcl/np.sum(wcl)
    meancl = [rng.uniform(size=xdim) for _ in range(ncl)]
    stdcl = [rng.uniform(size=xdim) for _ in range(ncl)]


    # linear constraints
    constraints = rng.uniform(size=(xdim, ydim))

    # sampling from prior
    train_x = sample_from_prior(probcl, meancl, stdcl,
                                nn_train + nn_test, seed)

    #labelling the training data
    train_y = np.matmul(train_x, constraints)
    nn_data = np.concatenate((train_x, train_y), axis=1)

    relunet = ReluNet(reludim, seed)
    relunet.train(nn_data[:nn_train])
    relunet.test(nn_data[nn_train:])    

    # fitting a model on a unlabelled dataset
    det_data = sample_from_prior(probcl, meancl, stdcl,
                                 det_train + det_valid, seed+1)
    feats = relunet.input_vars
    det = DET(feats, nmin, nmax)
    det.grow_full_tree(det_data[:det_train])
    det.prune_with_validation(det_data[det_train:])

    
    smt_relu = relunet.to_smt(threshold)
