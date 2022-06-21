# hybrid-benchmarks

Collection of benchmarks for *probabilistic inference* over **continuous**/**logical** variables with arbitrary **algebraic**/**logical** constraints.

## Structure-aware WMI-PA @ UAI22

Go to `uai-22/` and run:

`python3 generate_dets.py N_MIN N_MAX NQUERIES QUERYHARDNESS SEED`

where:

- `NMIN` and `NMAX` are constraints on the min. and max. number of instances in the leaves of the DET
- `NQUERIES` and `QUERYHARDNESS` respectively control how many queries over the learned models are generated and how many variables are involved in the query.
- `SEED` sets a seed number
