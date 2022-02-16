from det import DET
import numpy as np
import os
from pysmt.shortcuts import BOOL, LE, Plus, REAL, Real, Symbol, Times, reset_env
from pywmi import Density
from scipy.linalg import solve as solve_linear_system
from string import ascii_letters
from sys import argv



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

def read_feats(path):
    feats = []
    with open(featfile, 'r') as f:
        for line in f:
            if len(line) == 0:
                continue

            tokens = line.strip()[:-1].split(':')
            assert(len(tokens) > 0), "Couldn't parse any token"

            feat_name = "".join(c for c in tokens[0] if c in ascii_letters + '0123456789')
            str_type = tokens[1]                
                
            if str_type == "continuous" or str_type == "discrete":
                feats.append(Symbol(feat_name, REAL))

            elif str_type == "categorical":
                nvals = len(tokens[2].split(","))
                if nvals == 2:
                    feats.append(Symbol(feat_name, BOOL))
                else:
                    for i in range(nvals):
                        feats.append(Symbol(f'{feat_name}_OHE_{i}', BOOL))
            else:
                print("Unsupported feature type: {str_type}")
                exit(1)

    return feats


def read_data(path, feats):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if len(line) == 0:
                continue

            tokens = line[:-1].replace(' ','').split(',')                
            row = []
            i = 0
            for token in tokens:
                if 'OHE' not in feats[i].symbol_name():
                    if feats[i].symbol_type() == BOOL:
                        assert(token in ["0", "1"])
                        row.append(bool(int(token)))
                    else:
                        row.append(float(token))

                    i += 1

                else:
                    prefix = "_".join(
                        feats[i].symbol_name().split("_")[:-1])
                    cardinality = len([f for f in feats if prefix in
                                       f.symbol_name()])
                    assert(int(float(token)) < cardinality)
                    subrow = [False for _ in range(cardinality)]
                    subrow[int(float(token))] = True
                    row.extend(subrow)
                    i += cardinality

                data.append(row)

    return np.array(data)


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

    




                    

    
