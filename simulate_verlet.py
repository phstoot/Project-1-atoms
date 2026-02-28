import numpy as np
import matplotlib.pyplot as plt
from random import randint
import matplotlib.animation as animation
from functions import Lennard_Jones_Potential,\
    min_vector, \
    periodic_boundaries, \
    Interaction_force, \
    Kinetic_Energies


## Initial conditions
N = 2
L = 10
h = 0.001 # Natural units

# # for testing purposes:
pos = np.array([[-8, 5],
                [8, 5]], dtype=float)
vel = np.array([[20, 0],
                [-20,0]], dtype=float)


## Simulation function
# algorithmically, follow the following steps:
# 1) calculate positions at t+h with verlet position algorithm
# 2) calculate forces at t+h with potential function and positions
# 3) calculate velocities at t+h with verlet velocity algorithm

def verlet_integration():
    

    # we have N particles. For every particle i, we 
    # have to calculate interaction with particle j for j != i. Do the following:
    # Find force on a particle from other particle. Sum all forces to find resulting force,
    # then update positions from old velocities and update velocities with summed force
    
    # Try to avoid while loops and use for loops instead (less computations)
    # Only use for loops if necessary
    
    # this block calculates forces at time t
    F_t = np.zeros((N,2))
    
    for i in range(N):
        main_particle = pos[:,i]
        for j in range(N):
            interacting_particle = pos[:,j]
            if j != i:
                r_vector = min_vector(main_particle, interacting_particle)
                r = np.linalg.norm(r_vector)
                F_mag = -Interaction_force(r)
                U = Lennard_Jones_Potential(r)

                # Force vector, note that we already normalized the vector min_vec in the Force function definition
                F_t[i,0] += F_mag * r_vector[0]
                F_t[i,1] += F_mag * r_vector[1]
            
        # Due to for loop, i += 1 not necessary
    
    # Now comes the Verlet algorithm
    # (1) x(t+h) = x(t) + hv(t) + 0.5*h**2 * F(x(t))
    pos[:,0] += h*vel[:,0] + 0.5*(h**2)*F_t[:,0]
    pos[:,1] += h*vel[:,1] + 0.5*(h**2)*F_t[:,1]
    

    # (2) Calculate new forces with updated positions at time t+h:
    # note: Write this block into separate function since we use it often

    F_t_plus_h = np.zeros((N,2))
    for i in range(N):
        main_particle = pos[:,i]
        for j in range(N):
            interacting_particle = pos[:,j]
            if j != i:
                r_vector = min_vector(main_particle, interacting_particle)
                r = np.linalg.norm(r_vector)
                F_mag = -Interaction_force(r)
                U = Lennard_Jones_Potential(r)

                # Force vector, note that we already normalized the vector min_vec in the Force function definition
                F_t_plus_h[i,0] += F_mag * r_vector[0]
                F_t_plus_h[i,1] += F_mag * r_vector[1]

    # (3) update velocities
    vel[:,0] += 0.5*h*(F_t[:,0] + F_t_plus_h[:,0])
    vel[:,1] += 0.5*h*(F_t[:,1] + F_t_plus_h[:,1])

    # final detail: apply box constraints
    periodic_boundaries(pos)




colors = []
for i in range(N): # from some stackexchange post
    colors.append('#%06X' % randint(0, 0xFFFFFF))


# Animation part
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
ax.set_title("Verlet Algorithm")

# Create scatter object (this is what we update)
scat = ax.scatter(pos[:,0], pos[:,1], c=colors, s=80)

def update(frame):
    verlet_integration()  # advance system by one step
    scat.set_offsets(pos)  # update particle positions
    return scat,

ani = animation.FuncAnimation(
    fig,
    update,
    frames=400,
    interval=20,
    blit=True,
    repeat=False
)

plt.show()