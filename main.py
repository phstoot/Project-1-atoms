import numpy as np
import matplotlib.pyplot as plt

# Use astropy.units and astropy.constants? 


## Define variables (maybe useful for later)
N = 3
L = 10 # size of boxes, probably decide later

rho = ... # Particle density, not necessary for now
T = ... # temperature   
epsilon = 119.8 #K (epsilon / k_boltzmann)
sigma = 3.405 #Angstrom


## Equations of Motion

# Interaction potential (Lennard Jones)
def LennardJonesPotential(r):
    U = epsilon(
        (sigma / r)**12 - (sigma / r)**6
    )
    return U

#blablabla


## Define particles (init properties)
# Probably start with 3-4 particles first

# Initial Position
pos1 = [np.random([0,L]), np.random([0,L])]
pos2 = [np.random([0,L]), np.random([0,L])]
pos3 = [np.random([0,L]), np.random([0,L])]

print("Initial Positions: \n")
print("Position1: " +  str(pos1) + "\n" + "Position1: " +  str(pos2) + "\n" + "Position1: " +  str(pos3) + "\n")



## Find change due to interaction with neighbouring particles
# Simulate maybe 10 time steps first

h = 0.01
N_steps = 10






## Plot particles with their trajectories
