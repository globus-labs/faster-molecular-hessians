#! /bin/bash

xyz=../data/exact/caffeine_pm7_None.xyz
for step_size in 0.02 0.01 0.005; do
    # Do the randomized methods
    for method in 0_random-directions-same-distance.ipynb 1_random-directions-variable-distance.ipynb; do
        papermill -p starting_geometry $xyz -p step_size $s $method - > /dev/null
    done
    
    # Test with different reductions for "along axes"
    notebook=2_displace-along-axes.ipynb
    for n in 1 2 4; do
        papermill -p starting_geometry $xyz -p perturbs_per_evaluation $n $notebook - > /dev/null
    done
done
