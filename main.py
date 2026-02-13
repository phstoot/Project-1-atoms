import numpy as np
import matplotlib.pyplot as plt
# import astropy.units 
# import astropy.constants


# Use astropy.units and astropy.constants? --> sure, havent used these before but those might help? But astronomy units?


## Define variables (maybe useful for later)
N = 10
L = 10 # size of boxes, probably decide later - should we stick to Angstrom?

rho = ... # Particle density, not necessary for now
T = ... # temperature   
epsilon = 119.8 #K (epsilon / k_boltzmann)
sigma = 3.405 * 10**(-5)#Angstrom
mass = 10**(-27) # Mass

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

## Define particles (init properties)
# Probably start with 3-4 particles first

# Initial Position
# The more general approach comes from previous course
# vel = np.zeros((N,2)) - or are we choosing non-zero starting velocities?

pos = np.random.uniform(0,L,size=(N,2))
vel = np.zeros((N,2))

## Find change due to interaction with neighbouring particles
# Simulate maybe 10 time steps first

h = 0.1
N_steps = 10


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
    
    #Periodic boundary conditions
    pos[pos>L] -= L
    pos[pos<0] += L
    
    
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






