
from det import DET
import numpy as np
import numpy.random as rng
from pysmt.shortcuts import *
from relu import Dataset, ReluNet
import torch
from torch.utils.data import DataLoader


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

def train(dataloader, model, loss_fn, optimizer, device="cpu"):
    train_loss = 0
    i = 0
    for X, y in dataloader:
        
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()            
        i += 1

    train_loss /= len(dataloader)
    return train_loss


def test(dataloader, model, loss_fn, device="cpu"):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= len(dataloader)
    return test_loss


def run(train_dataloader, test_dataloader, max_epochs, model, loss_fn,
        optimizer, device):

    train_losses, test_losses = [], []
    for t in range(max_epochs):
        train_l = train(train_dataloader, model, loss_fn, optimizer, device)
        test_l = test(test_dataloader, model, loss_fn, device)
        train_losses.append(train_l)
        test_losses.append(test_l)
        if t % (max_epochs / 100) == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print("Train loss", train_l)
            print("Test loss", test_l)

    return train_losses, test_losses



if __name__ == '__main__':

    seed = 666

    # problem
    xdim = 3
    ydim = 2
    ncl = 2
    epsilon = 1e-2

    rng.seed(seed)

    print('Generating a Gaussian mixture as true prior P*(X)')
    wcl = rng.uniform(size=ncl)
    probcl = wcl/np.sum(wcl)
    meancl = [rng.uniform(size=xdim) for _ in range(ncl)]
    stdcl = [rng.uniform(size=xdim) for _ in range(ncl)]

    print('Generating linear constraints c: (Y = cX)')
    constraints = rng.uniform(size=(xdim, ydim))

    print('Training a Relunet on labelled data (Xt, Yt)')

    # relunet
    hiddendim = 32
    reludim = (xdim, hiddendim, ydim)
    nn_trainsize = 1000
    nn_testsize = 100
    nepochs = 100
    batch_size = 10
    lr = 1e-3
    loss = torch.nn.MSELoss()
    threshold = 0.0

    # sampling Xt ~ P*(X)
    sample = sample_from_prior(probcl, meancl, stdcl,
                               nn_trainsize + nn_testsize, seed)

    #labelling Yt = Xt * c
    train_x = sample[:nn_trainsize]
    test_x = sample[nn_trainsize:]
    train_y = np.matmul(train_x, constraints)
    test_y = np.matmul(test_x, constraints)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        rng.seed(worker_seed)
        random.seed(worker_seed) # dunno

    g = torch.Generator()
    g.manual_seed(0)

    train_dl = DataLoader(Dataset(train_x, train_y),
                          batch_size=batch_size,
                          worker_init_fn=seed_worker,
                          generator=g)
    test_dl = DataLoader(Dataset(test_x, test_y),
                         batch_size=batch_size,
                         worker_init_fn=seed_worker,
                         generator=g)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    relunet = ReluNet(reludim, seed)
    optimizer = torch.optim.SGD(relunet.parameters(), lr=lr)    
    train_losses, test_losses = run(train_dl,
                                    test_dl,
                                    nepochs,
                                    relunet,
                                    loss,
                                    optimizer,
                                    device)


    # encode the trained NN in SMT
    smt_relu = relunet.to_smt(threshold)

    print('Estimating P*(X) with DET trained on Xe ~ P*(X)')

    # Estimating P*(X)
    nmin = 10
    nmax = 50
    det_train = 5000
    det_valid = 500

    # fitting a model on a unlabelled dataset
    det_data = sample_from_prior(probcl, meancl, stdcl,
                                 det_train + det_valid, seed+1)
    feats = relunet.input_vars
    det = DET(feats, nmin, nmax)
    det.grow_full_tree(det_data[:det_train])
    det.prune_with_validation(det_data[det_train:])
    
    domain, smt_det, weight_det = det.to_pywmi()

    # generate queries
    queries = []
    for i, yi in enumerate(relunet.output_vars):
        ci = constraints[:,i]
        wc = Plus(*[Times(Real(float(ci[j])), xj)
                    for j,xj in enumerate(relunet.input_vars)])
        lower = LE(Plus(yi, Real(epsilon)), wc)
        upper = LE(wc, Minus(yi, Real(epsilon)))
        queries.append(And(lower, upper))
                   
        
    
    
