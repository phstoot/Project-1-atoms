import numpy as np
import matplotlib.pyplot as plt
from random import randint


## We could technically write all equations allowing for both natural and normal units (so using function(*,natural = True))


# Define variables
N = 4
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
    
    ## Check again if -13 or -14
    F = 24 * (2 * r**(-14) - r**(-8)
    )
    return F



## Minimal image convenction and periodic boundary conditions
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

def periodic_boundaries(pos):
    '''
    Causes the particles to stay in the box defined by (L,L).
    
    :param: pos (arr): Array of instantaneous positions of all particles.
    
    Returns: none
    '''
    for i in range(N):
        pos[i,0] %= L
        pos[i,1] %= L
        pos[pos < 0] += L


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

