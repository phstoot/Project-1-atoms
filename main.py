import numpy as np
import matplotlib.pyplot as plt

## Define variables (maybe useful for later)
N = 3
L = 10 # size of boxes, probably decide later

rho = ... # Particle density, not necessary for now
T = ... # temperature   



## Equations of Motion

# Interaction potential





## Define particles (init properties)
# Probably start with 2-3 particles first

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
