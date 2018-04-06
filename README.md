# Compute densities from a particle set

This is a simple code to compute Gaussian and tophat densities from particles in an N-body simulation. The code has limited use but it is very fast and simple to use and adapt. The Gaussian window is truncated at 3 sigmas (this can be easily changed in the code).

The code is based on a grid of pointers in which each cell is filled with the particles inside the cell. This grid is used to retrieve all particles around a given particle (using the 3x3x3 neighbor cells). This reduces the number of computations with grid size as ~ 27/N<sup>3</sup> (only true for a uniform particle distribution). This means that the code works optimally when the window used to compute densities (defined by the pointer grid) is much smaller than the size of the simulation box.

You can compute densities for all the particles in the set (not very useful as there are better options) and it can compute densities at sampling points defined by a sampling input file. Nowadays I use this code for the second option to compute densities at the position of haloes from dark matter particles and gas. The code can be easily modified to compute weighted averages of scalar values. This can be used to compute mean halo properties around all the haloes in a simulation.

The first version of code was written to compute densities and find sub-halos in this paper: http://adsabs.harvard.edu/abs/2007ApJ...655L...5A 
