
import numpy as np
import os
from pysmt.shortcuts import *
from pywmi import Density, Domain
from scipy.linalg import solve as solve_linear_system
from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
from spn.structure.Base import Context
from spn.structure.Base import Sum, Product
from spn.structure.StatisticalTypes import MetaType
from sys import argv
from utils import read_feats, read_data



def recw(node):
    if isinstance(node, Sum) or isinstance(node, Product):
        ch_res = [recw(c) for c in node.children]
        count = sum(res[1] for res in ch_res)

        if isinstance(node, Sum):
            w = Plus(*[Times(Real(node.weights[i]), res[0])
                       for i, res in enumerate(ch_res)])        
        else:
            w = Times(*[res[0] for res in ch_res])
    else:
        assert(len(node.scope) == 1)
        count = 1
        var = feats[node.scope[0]]

        print('---')
        print('leaf of type', var.symbol_type())
        print('histogram', node.densities)
        print('sum(histogram)', sum(node.densities))
        print('1-sum(histogram)', 1-sum(node.densities))
        print()

        if var.symbol_type() == BOOL:
            assert(len(node.densities) == 2)
            w = Ite(var,
                    Real(node.densities[1]),
                    Real(node.densities[0]))
        else:
            intervals = [And(LE(Real(node.breaks[i]), var),
                             LT(var, Real(node.breaks[i+1])))
                         for i in range(len(node.densities))]
            w = Ite(intervals[0],
                      Real(node.densities[0]),
                      Real(1-sum(node.densities)))
            for i in range(1, len(intervals)):
                w = Ite(intervals[i],
                          Real(node.densities[i]),
                          w)

    return w, count



if len(argv) != 8:
    print("Usage: python3 generate_mspns_syntetic.py N_VARS N_CLUSTERS MIN_INSTANCE_SLICES N_MODELS N_QUERIES QUERY_HARDNESS SEED")
    exit(1)

nvars, nclusters, mininstslices, nmodels, nqueries, qhardness, seed = int(argv[1]), int(argv[2]), int(argv[3]), int(argv[4]),\
    int(argv[5]), float(argv[6]), int(argv[7])

pwleaves = False

np.random.seed(seed)

for nm in range(nmodels):
    benchmark_folder = f'mspns-synth-{nvars}-{nclusters}-{mininstslices}-{nqueries}-{qhardness}-{seed}'

    if not os.path.isdir(benchmark_folder):
        os.mkdir(benchmark_folder)

    # fresh pysmt environment
    reset_env()

    mspnfile = os.path.join(benchmark_folder, f'{nm}.json')

    if os.path.isfile(mspnfile):
        print(f"{mspnfile} exists. Skipping.")
        continue


    trainxcluster = 100
    train = np.concatenate(tuple(np.random.normal(i*10, i, size=(trainxcluster, nvars)) for i in range(nclusters)))
    
    print(f"{nm} : Learning MSPN({mininstslices})")
    feats = [Symbol(f'X{i}', REAL) for i in range(nvars)]
    mtypes = [MetaType.REAL for _ in range(nvars)]
    ds_context = Context(meta_types=mtypes)
    ds_context.add_domains(train)
    leafop = create_piecewise_leaf if pwleaves else create_histogram_leaf
    mspn = learn_mspn(train, ds_context,
                      min_instances_slice=mininstslices,
                      leaves=leafop,
                      cpus=1)

    
    
    clauses = []
    bvars = []
    cvars = []
    cbounds = []
    for i, var in enumerate(feats):
        if var.symbol_type() == REAL:
            lb, ub = ds_context.domains[i]
            cvars.append(var.symbol_name())
            cbounds.append((lb, ub))
            clauses.append(LE(Real(float(lb)), var))
            clauses.append(LE(var, Real(float(ub))))
        else:
            bvars.append(var.symbol_name())

    support = And(*clauses)
    weight, nnodes = recw(mspn)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("NUMBER OF NODES:", nnodes)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    domain = Domain.make(bvars, cvars, cbounds)
        
    queries = []
    i = 0

    while i < nqueries:
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
        
        # consider a subset maybe?
        selected = np.random.choice(nvars, int(nvars*qhardness), replace=False)
        if len(selected) == 0:
            selected = [np.random.choice(nvars)]
            
        wx = [Times(Real(float(w[j])), x)
              for j,x in enumerate(domain.get_real_symbols())
              if j in selected]
        query = LE(Plus(*wx), Real(1))
        
        if is_sat(And(support, query)):
            queries.append(query)
            i += 1
        else:
            print(f"UNSAT {i+1}/{nqueries}")
            
    density = Density(domain, support, weight, queries)
    density.to_file(mspnfile)  # Save to file
    # density = Density.from_file(filename)  # Load from file




                    

    
