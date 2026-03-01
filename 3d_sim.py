import time
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import matplotlib.animation as animation
from functions import (
    Lennard_Jones_Potential,
    min_vector,
    Interaction_force,
    Kinetic_Energies,
)

timestr = time.strftime("%d-%m_%H.%M.%S")
## Initial conditions
N = 30
L = 10
h = 0.001  # Natural units

dim = 3
N_steps = 500

# # for testing purposes:
# pos = np.array([[-7, 5, 5],
#                 [7, 5, 5]], dtype=float)
# vel = np.array([[20, 0, 0],
#                 [-20,0, 0]], dtype=float)


pos = np.random.uniform(0, L, size=(N, dim))
vel = np.random.uniform(-4 * L, 4 * L, size=(N, dim))
kin_0 = np.sum(
    Kinetic_Energies(np.linalg.norm(vel, axis=1))
)  # initial kinetic energy to set y limit in animation


# Create position and velocity (3D) arrays to store data over time
All_pos = np.zeros((N, dim, N_steps + 1))
All_vel = np.zeros((N, dim, N_steps + 1))
All_kin = []
All_pot = []

for i in range(N):
    for j in range(dim):
        All_pos[i, j, 0] = pos[i, j]
        All_vel[i, j, 0] = vel[i, j]


## Simulation function
# algorithmically, follow the following steps:
# 1) calculate positions at t+h with verlet position algorithm
# 2) calculate forces at t+h with potential function and positions
# 3) calculate velocities at t+h with verlet velocity algorithm


def verlet_integration_3D(pos, vel):
    # we need to pass pos and vel otherwise it breaks. This is due to Python's distinction
    # between local and global variables. I think this is the point where we need
    # to start thinking about proper restructuring and creating classes.

    # we have N particles. For every particle i, we
    # have to calculate interaction with particle j for j != i. Do the following:
    # Find force on a particle from other particle. Sum all forces to find resulting force,
    # then update positions from old velocities and update velocities with summed force

    # Try to avoid while loops and use for loops instead (less computations)
    # Only use for loops if necessary

    # this block calculates forces at time t
    F_t = np.zeros((N, dim))
    U_t = 0

    for i in range(N):
        main_particle = pos[i, :]
        for j in range(N):
            interacting_particle = pos[j, :]
            if j != i:
                r_vector = min_vector(main_particle, interacting_particle, L=L, dim=dim)
                r = np.linalg.norm(r_vector)
                F_mag = -Interaction_force(r)
                U = Lennard_Jones_Potential(r)

                # Force vector, note that we already normalized the vector min_vec in the Force function definition
                F_t[i] += F_mag * r_vector

                U_t += (
                    0.5 * U
                )  # U is potential energy between pair, so for each particle one half
        # Due to for loop, i += 1 not necessary
    # store energies at time t
    kin = Kinetic_Energies(np.linalg.norm(vel, axis=1))
    All_kin.append(np.sum(kin))
    All_pot.append(U_t)

    # Now comes the Verlet algorithm to update our system
    # (1) x(t+h) = x(t) + hv(t) + 0.5*h**2 * F(x(t))
    pos += h * vel + 0.5 * (h**2) * F_t
    # apply box constraints
    pos %= L

    # (2) Calculate new forces with updated positions at time t+h:
    # note: Write this block into separate function since we use it often

    F_t_plus_h = np.zeros((N, dim))
    for i in range(N):
        main_particle = pos[i, :]
        for j in range(N):
            interacting_particle = pos[j, :]
            if i != j:
                r_vector = min_vector(main_particle, interacting_particle, L=L, dim=dim)
                r = np.linalg.norm(r_vector)
                F_mag = -Interaction_force(r)
                # Force vector, note that we already normalized the vector min_vec in the Force function definition
                F_t_plus_h[i] += F_mag * r_vector

    # (3) update velocities
    vel += 0.5 * h * (F_t + F_t_plus_h)


## Animate without storing data

# Animation part
fig = plt.figure(figsize=(5, 9))

# First Axes: the 3d scatter
ax = fig.add_subplot(2, 1, 1, projection="3d")
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_zlim(0, L)
ax.set_aspect("equal")
ax.set_title("Verlet Algorithm")
# Create scatter object (this is what we update)
scat = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])

# Second Axes: energy evolution
ax2 = fig.add_subplot(2, 1, 2)
(plot_kin,) = ax2.plot([], [], label="E_kin")
(plot_pot,) = ax2.plot([], [], label="E_pot")
# rolling window size
repeat_length = 500
ax2.set_xlim([0, repeat_length])
ax2.set_ylim([(0 - 0.1 * kin_0) / N, (kin_0 + 0.1 * kin_0) / N])
# ax2.set_yscale("symlog", linthresh=1e-2)
ax2.legend(loc=7)


def update(frame):
    verlet_integration_3D(pos, vel)  # advance system by one step
    scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    plot_kin.set_xdata(np.arange(frame))  # update time axis
    plot_kin.set_ydata(
        np.array(All_kin[0:frame]) / N
    )  # plot E/N to keep scale manageable
    plot_pot.set_xdata(np.arange(frame))
    plot_pot.set_ydata(np.array(All_pot[0:frame]) / N)
    if frame >= repeat_length:
        lim = ax2.set_xlim(frame - repeat_length, frame)
    else:
        lim = ax2.set_xlim(0, repeat_length)
    return scat, plot_kin


ani = animation.FuncAnimation(
    fig, update, frames=500, interval=20, blit=False, repeat=False
)
# writervideo = animation.PillowWriter(fps=50)
# ani.save(fr'Verlet_{N}_particles_{timestr}.gif')
plt.show()
