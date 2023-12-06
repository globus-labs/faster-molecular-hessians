#! /bin/bash

molecule=water
relax_method="b3lyp/cc-pvtz"

hess_methods="pm7/None xtb/None hf/cc-pvtz b3lyp/cc-pvtz wb97x-d/cc-pvtz m062x/cc-pvtz ccsd(t)/cc-pvdz"
deltas="0.04 0.02 0.01 0.005 0.0025"

#hess_methods="ccsd(t)/cc-pvtz"
#deltas=0.005

notebook=0_get-exact-answer.ipynb
for method in $hess_methods; do
    for delta in $deltas; do
        papermill -p hess_method $method -p relax_method $relax_method -p delta $delta -p molecule_name $molecule $notebook live.ipynb
    done
done
