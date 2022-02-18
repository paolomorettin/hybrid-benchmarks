
from det import DET
import numpy as np
import os
from pysmt.shortcuts import BOOL, LE, Plus, REAL, Real, Symbol, Times, reset_env
from pywmi import Density
from scipy.linalg import solve as solve_linear_system
from sys import argv
from utils import read_feats, read_data


def sample_uniform_hyperplane(polytope):
    nvars = len(polytope[0]) - 1

    # uniformly sampled point inside the polytope
    p = sample_uniform_point(polytope)

    # uniformly sampled orientation for the hyperplane
    o = np.random.uniform(0, 1, (nvars-1, nvars))

    # coefficients for the system of equations (i.e. n points in n dimensions)
    Points = p * np.concatenate((np.ones((1, nvars)), o))

    # solving the system to retrieve the hyperplane's coefficients
    # [p1 ; ... , pn] * coeffs = 1
    return solve_linear_system(Points, np.ones((nvars, 1))).transpose()


# full MLC suite sorted by increasing number of features
EXPERIMENTS = ['balance-scale', 'iris', 'cars', 'diabetes', 'breast-cancer', 'glass2', 'glass',
               'breast', 'solar', 'cleve', 'heart', 'australian', 'crx', 'hepatitis', 'german',
               'german-org', 'auto', 'anneal-U']

DATAFOLDER = 'mlc-datasets'



if len(argv) != 5:
    print("Usage: python3 generate_dets.py N_MIN N_MAX NQUERIES SEED")
    exit(1)

nmin, nmax, nqueries, seed = int(argv[1]), int(argv[2]), int(argv[3]), int(argv[4])

benchmark_folder = f'dets-{nmin}-{nmax}-{seed}'

if not os.path.isdir(benchmark_folder):
    os.mkdir(benchmark_folder)

for exp in EXPERIMENTS:

    # fresh pysmt environment
    reset_env()
    
    detfile = os.path.join(benchmark_folder, f'{exp}-{nmin}-{nmax}.json')

    if os.path.isfile(detfile):
        print(f"{detfile} exists. Skipping.")
        continue

    print(f"{exp} : Parsing data")    
    featfile = os.path.join(DATAFOLDER, f'{exp}.features')
    feats = read_feats(featfile)
    train = read_data(os.path.join(DATAFOLDER, f'{exp}.train.data'), feats)
    valid = read_data(os.path.join(DATAFOLDER, f'{exp}.valid.data'), feats)
    # we don't need this
    # test = read_data(os.path.join(DATAFOLDER, f'{exp}.test.data'), feats) 

    print(f"{exp} : Learning DET({nmin},{nmax})")
    det = DET(feats, nmin, nmax)
    det.grow_full_tree(train)
    det.prune_with_validation(valid)

    domain, support, weight = det.to_pywmi()    
    queries = []
    for i in range(nqueries):
        np.random.seed(seed+i)
        bbox = [domain.var_domains[v] for v in domain.real_vars]
        nvars = len(bbox)
        p = np.array([np.random.uniform(l, u) for l, u in bbox])
        # uniformly sampled orientation for the hyperplane
        o = np.random.uniform(0, 1, (nvars-1, nvars))
        # coefficients for the system of equations (i.e. n points in ndimensions)
        Points = p * np.concatenate((np.ones((1, nvars)), o))

        # solving the system to retrieve the hyperplane's coefficients
        # [p1 ; ... , pn] * coeffs = 1
        w = solve_linear_system(Points, np.ones((nvars, 1))).transpose()[0]
        wx = [Times(Real(float(w[j])), x)
              for j,x in enumerate(domain.get_real_symbols())]
        query = LE(Plus(*wx), Real(1))
        queries.append(query)
    
    density = Density(domain, support, weight, queries)

    density.to_file(detfile)  # Save to file
    # density = Density.from_file(filename)  # Load from file

    




                    

    
