import numpy as np
import matplotlib.pyplot as plt
from random import randint


# Define variables
N = 3
L = 10 # size of box, probably decide later

rho = ... # Particle density, not necessary for now
T = ... # temperature   
epsilon = 1.654 * 10**(-10) #119.8 #K (epsilon / k_boltzmann)
sigma = 3.405 * 10**(-10) # Angstrom
mass = 6.6 * 10**(-26) # Mass

## Equations of Motion
# Force from interaction
def Interaction_force(r):
    F = 24 * epsilon * (2 * 
        (sigma**12 / (r**14)) - (sigma**6 / (r**8))
    )
    return F



# distance, duh - should we already calculate the minimal distance here? - actually calculated minimal vector here (so vector )
def min_vector(part1, part2):
    '''Finds smallest vector connecting particle 1 to particle 2, in the smallest image convention.'''
    vec = part2 - part1
    min_vec = np.mod(vec + [0.5*L, 0.5*L], [L,L]) - [0.5*L, 0.5*L]
    return min_vec

def periodic_boundaries(pos):
    '''Causes the particles to stay in the box defined by (L,L).'''
    for i in range(N):
        pos[i,0] %= L
        pos[i,1] %= L
        pos[pos < 0] += L



## Define particles (init properties)
# Probably start with 3-4 particles first

# shall we create a particle class?
class Particle:
    def __init__(self, position, velocity, mass):
        self.position = position
        self.velocity = velocity
        self.mass = mass

# Initial Position
pos = np.random.uniform(0,L,size=(N,2))
vel = np.random.uniform(0,L/5, size=(N,2))# np.zeros((N,2))

## Find change due to interaction with neighbouring particles
# Simulate maybe 10 time steps first

h = 0.01
N_steps = 50


print("Old Positions: "+ str(pos))
print("Old velocities: "+ str(vel))


## Create position and velocity (3D) arrays to store data
All_pos = np.zeros((N,2,N_steps+1))
All_vel = np.zeros((N,2,N_steps+1))

for i in range(N):
    All_pos[i,0,0] = pos[i,0]
    All_pos[i,1,0] = pos[i,1]
    All_vel[i,0,0] = vel[i,0]
    All_vel[i,1,0] = vel[i,1]

# print(str(All_pos))
# print(str(All_vel))



## Simulation part
def simulate():
    '''
    For each particle: Finds force from every other interaction particle (using the smallest vector combining them). 
    Adds the forces up in an array, such that the final force is the sum of all.
    
    Then modifies position and velocities from that point on. 
    
    Later on, this is supposed to run N_steps times.
    '''
    
    
    # I guess the idea here is the following: Find force onto each particle from other particles. New velocity is then result of sum of forces.
    # Additionally, slightly change position from old velocity.
    # Overwrite pos & vel of each particle
    Summed_Force = np.zeros((N,2))
    
    j = 0
    while j < N:      
        main_particle = pos[j,:]
        
        k = 0
        while k < N:
            int_particle = pos[k,:]
            if k != j:
                # Vector connecting main to interaction particle
                min_vec = min_vector(main_particle, int_particle)
                r = np.linalg.norm(min_vec)
                
                # Force magnetiude
                F_mag = Interaction_force(r)
                
                # Force vector, note that we already normalized the vector min_vec in the Force function definition
                Summed_Force[k,0] += F_mag * min_vec[0]
                Summed_Force[k,1] += F_mag * min_vec[1]

            k += 1
        
        j += 1           
    
    
    ## After calculation
    pos[:,0] += vel[:,0] * h
    pos[:,1] += vel[:,1] * h

    vel[:,0] += 1 / mass * Summed_Force[:,0] * h
    vel[:,1] += 1 / mass * Summed_Force[:,1] * h
    
    #Periodic boundary conditions (Needs to be adjusted, so that if a particle with pos > 2L is floored to its remainder between (0,L))
    periodic_boundaries(pos)
    
    
    print("New Positions: "+ str(pos))
    print("New velocities: "+ str(vel))

    ## Store pos and velocity in overall array for plotting the evolution later o
    
    return pos, vel
    
    
    
 
for k in range (N_steps):
    simulate()
    ## Store pos and velocity in overall array for plotting the evolution later on
    for m in range(N):
        All_pos[m,0,k+1] = pos[m,0]
        All_pos[m,1,k+1] = pos[m,1]
        All_vel[m,0,k+1] = vel[m,0]
        All_vel[m,1,k+1] = vel[m,1]

print(str(All_pos))
print(str(All_vel))
## Plot particles with their trajectories



## Regular Evolution Plot
def static_plot(positions, velocities):
    '''
    Plots the motion of the particles on one plot.
    
    '''   
    
    
    # Define colours for plot
    colors = []
    for i in range(N): # from some stackexchange post
        colors.append('#%06X' % randint(0, 0xFFFFFF))
        
    plt.figure(figsize=(8,8))    
    
    for i in range(N):
    plt.scatter(positions[i,0,:], positions[i,1,:], c= colors[i], marker = 'o', linewidths= 1)


    plt.title(r"Simulation for " + str(N) + " different particles")
    plt.xlabel(r"x")
    plt.ylabel(r"y")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# ## Active animation
# def animated_plot():
#     .

static_plot(All_pos, All_vel)





## Things still to do: 
# Make a more understandable plot (especially when particles go haywire, their evolution cannot be captured in the plot)
# Adjust parameters to make it physical
# Maybe mark starting locations in plot
# Can be plotted continously using FuncAnimation??
# Check validity: Do collisions reproduce our expectation? Single particle moves in straight line?
# 

