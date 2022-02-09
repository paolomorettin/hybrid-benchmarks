
from numpy import inf
import numpy.random as rng
import os
from pysmt.shortcuts import And, Bool, Real, LE, reset_env
from pywmi import Density, Domain
from relu import ReluNet
from sys import argv

if len(argv) <= 1:
    print("USAGE: python3 generate_random.py SEED MODE N-PROBLEMS DIMENSIONS")
    exit(1)

seed = int(argv[1])
mode = argv[2].lower().strip()
nproblems = int(argv[3])
dimensions = [int(x) for x in argv[4:]]

benchmark_folder = f'relu-{mode}-{seed}-'+'-'.join(map(str, dimensions))

if not os.path.isdir(benchmark_folder):
    os.mkdir(benchmark_folder)

for i in range(nproblems):

    print(f"Generating [{i+1}/{nproblems}] {mode} problem (seed {seed}) w/ dim {dimensions}")

    # fresh pysmt environment
    reset_env()

    relufile = os.path.join(benchmark_folder, f'relu-{i}.json')

    nn = ReluNet(dimensions, seed+i)
    formula = nn.formula

    rng.seed(seed+i)
    if mode == 'uniform':
        cnames = []
        bounds= []        
        smtbounds = []
        for var in nn.input_vars:
            l, u = sorted([rng.uniform(), rng.uniform()])
            assert(l <= u)
            smtbounds.append(LE(Real(l), var))
            smtbounds.append(LE(var, Real(u)))
            cnames.append(var.symbol_name())
            bounds.append((-inf,+inf))

        cnames.extend([v.symbol_name()
                       for v in formula.get_free_variables()
                       if v.symbol_name() not in cnames])
        bounds.extend([(-inf, +inf) for _ in range(len(cnames)-len(bounds))])
        domain = Domain.make([], cnames, bounds)
        support = And(formula, *smtbounds)
        weight = Real(1)
    
    else:
        print(f"Unsupported MODE: '{mode}'")
        exit(1)        


    queries = [] # No queries for now
    density = Density(domain, support, weight, queries)
    density.to_file(relufile)  # Save to file



