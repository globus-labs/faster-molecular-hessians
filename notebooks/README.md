# Method Development

The notebooks here explore variations of approximate methods for computing Hessians.

Our analysis starts with getting the exact answer and storing it in the `data` subfolder in this directory.

From there, we...

1. Compute initial training sets with different sampling strategies
2. Explore methods for fitting approximate models

## Notebook Structure

All notebooks start with configuration cells that define the molecule and accuracy level
being studied (e.g., caffeine at B3LYP//def2-svp) then any options for the methods described
in the notebooks.
These configuration settings are used to define the names of the outputs, 
which are stored in a subdirectory named "data."
