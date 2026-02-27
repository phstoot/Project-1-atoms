# This file is for testing some stuff out, 
# such as plotting the potential as a function of the radius, 
# or playing around with functions and variables.

import numpy as np
from matplotlib import pyplot as plt


from functions import Lennard_Jones_Potential,\
    min_vector, periodic_boundaries

radii_small = np.linspace(10**-3,1,1000, dtype=float)
radii_large = np.linspace(1,20,1000, dtype=float)

potentials_small = Lennard_Jones_Potential(radii_small)
potentials_large = Lennard_Jones_Potential(radii_large)



fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(radii_small, potentials_small)
ax2.plot(radii_large, potentials_large)
ax1.set_yscale("log")
plt.show()