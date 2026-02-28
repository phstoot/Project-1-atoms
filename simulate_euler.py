import numpy as np
import matplotlib.pyplot as plt
from random import randint
import matplotlib.animation as animation
from functions import Lennard_Jones_Potential,\
    min_vector, \
    periodic_boundaries, \
    Interaction_force, \
    Kinetic_Energies

## We could technically write all equations allowing for both natural and normal units (so using function(*,natural = True))
# Define variables !!! are both used in functions file and here. Find a way to restructure this
N = 2
L = 10 # size of box, probably decide later

rho = ... # Particle density, not necessary for now
T = ... # temperature   
epsilon = 1.654 * 10**(-10) #119.8 #K (epsilon / k_boltzmann)
sigma = 3.405 * 10**(-10) # Angstrom
mass = 6.6 * 10**(-26) # Mass


## Define particles (init properties)
# Probably start with 3-4 particles first

# shall we create a particle class?
class Particle:
    def __init__(self, position, velocity, mass):
        self.position = position
        self.velocity = velocity
        self.mass = mass

# Initialisation
# pos = np.random.uniform(0,L,size=(N,2))
# vel = np.random.uniform(-L,L, size=(N,2)) # np.zeros((N,2)) #
# kin = Kinetic_Energies(np.linalg.norm(vel, axis=1))

# # for testing purposes:
pos = np.array([[-6, 5.2],
                [6, 5]], dtype=float)
vel = np.array([[10, 0],
                [-10,0]], dtype=float)
kin = Kinetic_Energies(np.linalg.norm(vel, axis=1))

## Find change due to interaction with neighbouring particles
# Simulate maybe 10 time steps first

h = 0.001 # Due to our redefinition
N_steps = 200

print("Start positions: "+ str(pos))
print("Start velocities: "+ str(vel))


## Create position and velocity (3D) arrays to store data over time
All_pos = np.zeros((N,2,N_steps+1))
All_vel = np.zeros((N,2,N_steps+1))
All_kin = []
All_pot = []

for i in range(N):
    All_pos[i,0,0] = pos[i,0]
    All_pos[i,1,0] = pos[i,1]
    All_vel[i,0,0] = vel[i,0]
    All_vel[i,1,0] = vel[i,1]


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
                
                # Force magnitude
                F_mag = -Interaction_force(r)

                # Potential energy from pair of particles
                U = Lennard_Jones_Potential(r)
                
                # Force vector, note that we already normalized the vector min_vec in the Force function definition
                Summed_Force[j,0] += F_mag * min_vec[0]
                Summed_Force[j,1] += F_mag * min_vec[1]

            k += 1
        
        j += 1           
    
    
    ## After calculation
    pos[:,0] += vel[:,0] * h
    pos[:,1] += vel[:,1] * h

    vel[:,0] += Summed_Force[:,0] * h
    vel[:,1] += Summed_Force[:,1] * h

    kin = Kinetic_Energies(np.linalg.norm(vel, axis=1))
    All_kin.append(np.sum(kin))
    All_pot.append(U)

    
    #Periodic boundary conditions (Needs to be adjusted, so that if a particle with pos > 2L is floored to its remainder between (0,L))
    periodic_boundaries(pos)

    return pos
    
    
    


# ## Define starting position and velocity here (will be put in a separate run file later I guess)
# N = 2

# h = 0.00001 # Due to our redefinition
# N_steps = 5
# pos = np.array([[0.3*L, ], [2*L/3,L/2+0.001]])
# vel = np.array([[L,0],[-L/2,0]])#np.zeros((N,2)) #



# First try of simulation, without function but in loop to store data

for k in range (N_steps):
    simulate()
    ## Store pos and velocity in overall array for plotting the evolution later on
    for m in range(N):
        All_pos[m,0,k+1] = pos[m,0]
        All_pos[m,1,k+1] = pos[m,1]
        All_vel[m,0,k+1] = vel[m,0]
        All_vel[m,1,k+1] = vel[m,1]
        

## Plot particles with their trajectories


colors = []
for i in range(N): # from some stackexchange post
    colors.append('#%06X' % randint(0, 0xFFFFFF))

## Regular Evolution Plot
def static_plot(positions, velocities):
    '''
    Plots the motion of the particles on one plot.
    
    '''   
    
    
    # Define colours for plot
        
    plt.figure(figsize=(8,8))    
    
    for i in range(N):
        plt.plot(positions[i,0,:], positions[i,1,:], c= colors[i], linestyle = "-", linewidth= 1)
        plt.scatter(positions[i,0,N_steps], positions[i,1,N_steps], c=colors[i], marker= "o") 


    plt.title(r"Simulation for " + str(N) + " different particles")
    plt.xlabel(r"x")
    plt.ylabel(r"y")
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

Total_energy = np.array(All_kin) + np.array(All_pot)


# # Plot the potential and kinetic energies of system over time
plt.figure(figsize=(6,3))
plt.plot(np.arange(N_steps), All_kin, label='Ekin')
plt.plot(np.arange(N_steps), All_pot, label='Epot')
plt.plot(np.arange(N_steps), Total_energy, label='Etot')
plt.legend()
plt.xlabel('t')
plt.ylabel('E')
plt.show()






# # Animation part
# fig, ax = plt.subplots(figsize=(6,6))
# ax.set_xlim(0, L)
# ax.set_ylim(0, L)
# ax.set_aspect('equal')
# ax.set_title("Particle Simulation")

# # Create scatter object (this is what we update)
# scat = ax.scatter(pos[:,0], pos[:,1], c=colors, s=80)

# def update(frame):
#     simulate()  # advance system by one step
#     scat.set_offsets(pos)  # update particle positions
#     return scat,

# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=200,
#     interval=20,
#     blit=True,
#     repeat=False
# )

# plt.show()

# t = np.linspace(0,1000,1001)
# plt.plot(t, np.linalg.norm(All_vel, axis=1)[0])
# plt.plot(t, np.linalg.norm(All_vel, axis=1)[1])
# plt.show()



# def update(frame):
#     # for each frame, update the data stored on each artist.
#     pos, vel = simulate()
#     # update the scatter plot:
#     for i in range(N):
#         ax.plot(pos[i,0], pos[i,1], c= colors[i], linestyle = "-", linewidth= 1)
#         ax.scatter(pos[i,0], pos[i,1], c=colors[i], marker= "o")
    
#     # data = np.stack([x, y]).T
#     # scat.set_offsets(data)
#     # # update the line plot:
#     # line2.set_xdata(t[:frame])
#     # line2.set_ydata(z2[:frame])
#     return (scat, line2)



# myAnimation = animation.FuncAnimation(fig,simulate,np.arange(1, 20),interval=1, blit=True, repeat=False)
# myAnimationvel = animation.FuncAnimation(fig2,simulate,np.arange(1, 2000),interval=1, blit=True, repeat=False)
# plt.show()







## Things still to do: 
# Make a more understandable plot (especially when particles go haywire, their evolution cannot be captured in the plot)
# Adjust parameters to make it physical
# Maybe mark starting locations in plot
# Can be plotted continously using FuncAnimation??
# Check validity: Do collisions reproduce our expectation? Single particle moves in straight line?
# Put the motivations written in this code into latex
# Save energy at each instant
