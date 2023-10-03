# Explore Sampling Methods

Our approximate methods use a training set of molecular energies in slightly different conformations.
Each notebook in this directory implements a different strategy for sampling conformations.
All save ASE databases with the data into a different subdirectory of `data`, 
with each db named after the input geometry and any sampling settings. 
Terms in the database names are separated by underscores.
