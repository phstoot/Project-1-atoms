# This file is for testing some stuff out, 
# such as plotting the potential as a function of the radius, 
# or playing around with functions and variables.

import numpy as np
from matplotlib import pyplot as plt


from functions import Lennard_Jones_Potential, Interaction_force, \
    min_vector, periodic_boundaries

radii_small = np.linspace(10**-3,1,1000, dtype=float)
radii_large = np.linspace(1,2.5,1000, dtype=float)

potentials_small = Lennard_Jones_Potential(radii_small)
potentials_large = Lennard_Jones_Potential(radii_large)
force_small = Interaction_force(radii_small)
force_large = Interaction_force(radii_large)


fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(radii_small, potentials_small)
ax2.plot(radii_large, potentials_large)
ax1.set_yscale("log")
plt.show()



## test while loop vs for loop
N = 1000

k = 0
while k < N:      
    m = 0
    while m < N:
        if m != k:
             print('while loop', k, m)
        m += 1
        
    k += 1 

N = 1000

for i in range(N):
        for j in range(N):
            if i != j:
                print('for loop:', i, j)

# N = 100:
# While loop:
# CPU times: user 60.2 ms, sys: 12.1 ms, total: 72.2 ms
# Wall time: 72.2 ms
# For loop:
# CPU times: user 60.8 ms, sys: 11.1 ms, total: 71.9 ms
# Wall time: 71.3 ms

# N = 1000:
# While loop:
# CPU times: user 5.37 s, sys: 871 ms, total: 6.24 s
# Wall time: 6.53 s
# For loop:
# CPU times: user 5.35 s, sys: 879 ms, total: 6.23 s
# Wall time: 6.5 s


