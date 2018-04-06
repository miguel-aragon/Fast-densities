# Compute densities from a particle set

This is a simple code to compute Gaussian and tophat densities from particles in an N-body simulation. The code has limited use but it is very fast and simple to use and adapt.

It works best when the window used to compute densities is much smaller than the size of the simulation box.

You can compute densities using Gaussian and tophat windows. The Gaussian window is truncated at 3 sigmas (this can be easily changed in the code).

The code can compute densities for all the particles in the set (not very useful as there are better options) and it can compute densities at sampling points defined at an sampling input file. Nowadays I use this code for the second option to compute densities at the position of haloes from dark matter particles and gas. The code can be easily modified to compute weighted averages of scalar values. This can be used to compute mean halo properties around all the haloes in a simulation.

