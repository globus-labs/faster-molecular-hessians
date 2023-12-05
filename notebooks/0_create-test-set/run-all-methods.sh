#! /bin/bash

molecule=butanol
methods="pm7//None xtb//None hf//cc-pvtz b3lyp//cc-pvtz wb97x-d//cc-pvtz m062x//cc-pvtz ccsd(t)//cc-pvdz"
deltas="0.04 0.02 0.01 0.005 0.0025"

#methods="ccsd(t)//cc-pvtz"
#deltas=0.005

notebook=0_get-exact-answer.ipynb
for name in $methods; do
    echo $name
    for delta in $deltas; do
        method=$(echo $name | cut -d "/" -f 1)
        basis=$(echo $name | cut -d "/" -f 3)
        papermill -p method $method -p basis $basis -p delta $delta -p molecule_name $molecule $notebook live.ipynb
    done
done
