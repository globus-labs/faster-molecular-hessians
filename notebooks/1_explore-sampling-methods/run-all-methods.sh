#! /bin/bash

xyz=../data/exact/caffeine_pm7_None.xyz
method='pm7/None'

for step_size in 0.02; do
    # Do the randomized methods
    for notebook in 0_random-directions-same-distance.ipynb 1_random-directions-variable-distance.ipynb; do
        papermill -p starting_geometry $xyz -p method $method -p step_size $step_size $notebook last.ipynb
    done
    
    # Test with different reductions for "along axes"
    notebook=2_displace-along-axes.ipynb
    for n in 2 4 8; do
        papermill -p starting_geometry $xyz -p method $method -p perturbs_per_evaluation $n -p step_size $step_size $notebook last.ipynb
    done
done

# Test with the vibrational modes
notebook=3_displace-along-vibrational-modes.ipynb
for step_size in 0.001 0.002; do  # These step sizes are energy scales in eV, not distances in Angstrom as above
    for n in 16 32 64; do
        papermill -p starting_geometry $xyz -p method $method -p perturbs_per_evaluation $n -p step_size $step_size $notebook last.ipynb
    done
done
