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
N = 3
L = 10
h = 0.001 # Natural units

N_steps = 500

# # for testing purposes:
# pos = np.array([[-6.5, 5.1],
#                 [6.5, 5]], dtype=float)
# vel = np.array([[8, 0],
#                 [-8,0]], dtype=float)

pos = np.random.uniform(0,L,size=(N,2))
vel = np.random.uniform(-4*L,4*L, size=(N,2))
kin_0 = np.sum(Kinetic_Energies(np.linalg.norm(vel, axis=1))) # initial kinetic energy to set y limit in animation


#Create position and velocity (3D) arrays to store data over time
All_pos = np.zeros((N,2,N_steps+1))
All_vel = np.zeros((N,2,N_steps+1))
All_kin = []
All_pot = []

for i in range(N):
    All_pos[i,0,0] = pos[i,0]
    All_pos[i,1,0] = pos[i,1]
    All_vel[i,0,0] = vel[i,0]
    All_vel[i,1,0] = vel[i,1]



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
        main_particle = pos[i,:]
        for j in range(N):
            interacting_particle = pos[j,:]
            if j != i:
                r_vector = min_vector(main_particle, interacting_particle)
                r = np.linalg.norm(r_vector)
                F_mag = -Interaction_force(r)
                U = Lennard_Jones_Potential(r)

                # Force vector, note that we already normalized the vector min_vec in the Force function definition
                F_t[i,0] += F_mag * r_vector[0]
                F_t[i,1] += F_mag * r_vector[1]
                
                All_pot.append(0.5*U) # U is potential energy between pair, so for each particle one half
        # Due to for loop, i += 1 not necessary
    # store energies at time t
    kin = Kinetic_Energies(np.linalg.norm(vel, axis=1))
    All_kin.append(np.sum(kin))
    

    # Now comes the Verlet algorithm to update our system
    # (1) x(t+h) = x(t) + hv(t) + 0.5*h**2 * F(x(t))
    pos[:,0] += h*vel[:,0] + 0.5*(h**2)*F_t[:,0]
    pos[:,1] += h*vel[:,1] + 0.5*(h**2)*F_t[:,1]
    # apply box constraints
    periodic_boundaries(pos)

    # (2) Calculate new forces with updated positions at time t+h:
    # note: Write this block into separate function since we use it often

    F_t_plus_h = np.zeros((N,2))
    for i in range(N):
        main_particle = pos[i,:]
        for j in range(N):
            interacting_particle = pos[j,:]
            if i != j:
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



    

## simulate over time with storage for plotting

# for k in range (N_steps):
#     verlet_integration()
#     ## Store pos and velocity in overall array for plotting the evolution later on
#     for m in range(N):
#         All_pos[m,0,k+1] = pos[m,0]
#         All_pos[m,1,k+1] = pos[m,1]
#         All_vel[m,0,k+1] = vel[m,0]
#         All_vel[m,1,k+1] = vel[m,1]

# Total_energy = np.array(All_kin) + np.array(All_pot)


# # # Plot the potential and kinetic energies of system over time
# plt.figure(figsize=(12,3))
# plt.plot(np.arange(150), All_kin[75:225], label='Ekin')
# plt.plot(np.arange(150), All_pot[75:225], label='Epot')
# plt.plot(np.arange(150), Total_energy[75:225], label='Etot')
# plt.legend()
# plt.title('Energy evolution')
# plt.xlabel('t')
# plt.ylabel('E')
# plt.show()


# plt.figure(figsize=(6,6))
# for i in range(N):
#     plt.plot(All_pos[i,0,:], All_pos[i,1,:])
# plt.title('Particle trails')
# plt.show()






## Animate without storing data

colors = []
for i in range(N): # from some stackexchange post
    colors.append('#%06X' % randint(0, 0xFFFFFF))


# Animation part
fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6,8), height_ratios=[6,3])
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
ax.set_title("Verlet Algorithm")

# Create scatter object (this is what we update)
scat = ax.scatter(pos[:,0], pos[:,1], c='r', s=80)
plot_kin, = ax2.plot([],[], label='E_kin')
plot_pot, = ax2.plot([],[], label='E_pot')
#rolling window size
repeat_length=500
ax2.set_xlim([0,repeat_length])
ax2.set_ylim([(0 - 0.1*kin_0), (kin_0 + 0.1*kin_0)])
ax2.legend()

def update(frame):
    verlet_integration()  # advance system by one step
    scat.set_offsets(pos)  # update particle positions
    plot_kin.set_xdata(np.arange(frame)) # update time axis
    plot_kin.set_ydata(All_kin[0:frame]) # update energy data
    plot_pot.set_xdata(np.arange(frame))
    plot_pot.set_ydata(All_pot[0:frame])
    if frame >= repeat_length:
        lim = ax2.set_xlim(frame-repeat_length, frame)
    else:
        lim = ax2.set_xlim(0, repeat_length)
    return scat, plot_kin

ani = animation.FuncAnimation(
    fig,
    update,
    frames=500,
    interval=20,
    blit=False,
    repeat=False
)

plt.show()