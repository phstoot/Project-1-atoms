import numpy as np
import matplotlib.pyplot as plt
from random import randint
from numba import njit


## We could technically write all equations allowing for both natural and normal units (so using function(*,natural = True))


# Define variables
N = 2
L = 10 # size of box, probably decide later

rho = ... # Particle density, not necessary for now
T = ... # temperature   
epsilon = 1.654 * 10**(-10) #119.8 #K (epsilon / k_boltzmann)
sigma = 3.405 * 10**(-10) # Angstrom
mass = 6.6 * 10**(-26) # Mass

def Interaction_force(r):
    '''
    Calculates force in natural units. 
        
    :param r (float): Distance between two particles in natural units.
    
    :return (float) : Magnitude of Interaction Force  
    '''
    
    
    F = 24 * (2 * r**(-14) - r**(-8)
    )
    return F

## Minimal image convenction
def min_vector(part1, part2):
    '''Finds smallest vector connecting particle 1 to particle 2, in the smallest image convention.
    
    Parameters:
    part1 (arr): main particle
    part2 (arr): interaction particle
    
    Returns:
    arr: Vector pointing to closest version of interaction particle.
    '''
    vec = part2 - part1
    min_vec = np.mod(vec + [0.5*L, 0.5*L], [L,L]) - [0.5*L, 0.5*L]
    return min_vec

## Energies
def Kinetic_Energies(vel):
    '''
    Calculates the Kinetic Energy of each particle in natural units.
    
    :param vel (arr): Instantaneous velocity array in natural units.
    
    :return (float): Kinetic Energy array in natural units.
    '''
    
    Kin = 1/2 * vel * vel
    return Kin
     
def Lennard_Jones_Potential(r):
    '''
    Calculates Potential Energy due to each particle interaction, in natural units. 
        
    :param r (float): Distance between two particles in natural units.
    
    :return (float): Interaction Potential
    '''
    
    U = 4 * (
        r**(-12) - r**(-6)
    )
    return U

@njit
def compute_forces_numba(positions, boxsize, rcut, 
                        cell_particles, cell_counts, 
                        num_cells_per_dim):

    N = positions.shape[0]
    forces = np.zeros_like(positions)

    for cx in range(num_cells_per_dim):
        for cy in range(num_cells_per_dim):
            for cz in range(num_cells_per_dim):

                cell_id = cx + num_cells_per_dim*(cy + num_cells_per_dim*cz)

                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):

                            nx = (cx + dx) % num_cells_per_dim
                            ny = (cy + dy) % num_cells_per_dim
                            nz = (cz + dz) % num_cells_per_dim

                            neighbor_id = nx + num_cells_per_dim*(ny + num_cells_per_dim*nz)

                            for a in range(cell_counts[cell_id]):
                                i = cell_particles[cell_id, a]
                                if i == -1:
                                    continue

                                for b in range(cell_counts[neighbor_id]):
                                    j = cell_particles[neighbor_id, b]

                                    if j <= i:
                                        continue

                                    rij = positions[j] - positions[i]

                                    # minimal image
                                    for k in range(3):
                                        if rij[k] > 0.5 * boxsize:
                                            rij[k] -= boxsize
                                        elif rij[k] < -0.5 * boxsize:
                                            rij[k] += boxsize

                                    dist2 = rij[0]**2 + rij[1]**2 + rij[2]**2

                                    if dist2 < rcut * rcut:
                                        dist = np.sqrt(dist2)
                                        fmag = -Interaction_force(dist)

                                        for k in range(3):
                                            f = fmag * rij[k]
                                            forces[i, k] += f
                                            forces[j, k] -= f

    return forces