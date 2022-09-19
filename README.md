
A collection of benchmarks for *PROBABILISTIC INFERENCE* 
with **algebraic** and **logical** constraints.


## Hybrid UCI datasets

Some benchmarks are based on (a selection of) [UCI
datasets](https://archive.ics.uci.edu/ml/index.php) having both
continuous and discrete features. These datasets are included in
`data/uci-datasets`.


## Random queries on DETs trained on UCI datasets

The paper ["SMT-based weighted model integration with structure
awareness"](https://proceedings.mlr.press/v180/spallitta22a/spallitta22a.pdf)
(Spallitta et al., 2022) first featured experiments where ["Density
Estimation Trees"](https://dl.acm.org/doi/pdf/10.1145/2020408.2020507)
(Ram and Grey, 2011) are trained on UCI datasets.

A number of random queries over the continuous variable models are
generated:

$Pr(w X \le b)$ with $w \in \mathbb{R}^k, b \in \mathbb{R}$

Go to `uci-det/` and run:

`python3 generate_dets.py N_MIN N_MAX NQUERIES QUERYHARDNESS SEED`

where:

- `NMIN` and `NMAX` are hyperparameters of the greedy DET learning algorithm. They constrain the min. and max. number of instances in the leaves of the DET
- `NQUERIES` and `QUERYHARDNESS` respectively control how many queries over the learned models are generated and the ratio of variables involved in the query.
- `SEED` sets a seed number


## Random queries on SPNs trained on UCI datasets

The same setting was investigated, using Sum-Product Networks (see
e.g. ["Sum-product networks: A new deep
architecture"](https://ieeexplore.ieee.org/iel5/6114268/6130192/06130310.pdf)
(Poon and Domingos, 2011)) instead of DETs.

Go to `uci-spn/` and run:

`python3 generate_spns.py N_MIN N_MAX NQUERIES QUERYHARDNESS SEED`

where:

- `NMIN` and `NMAX` are hyperparameters of the greedy DET learning algorithm. They constrain the min. and max. number of instances in the leaves of the DET
- `NQUERIES` and `QUERYHARDNESS` respectively control how many queries over the learned models are generated and the ratio of variables involved in the query.
- `SEED` sets a seed number

