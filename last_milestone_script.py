from simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from functions import lennard_jones_potential, interaction_force, min_vector

# To do: run set of simulations, to sample both different configurations at different times and with different initial conditions
# 1) Equilibrate
# 2) Run:
# 3) Sample different configurations at several t, only store the important variables (and store those we need for later externally)
# 4) Reset simulation
# Note: This is averaging is done
# for each independant bin [r,r+ ∆r]

test = Simulation(density=0.3, temp=5) 

# set_counts = []

# for i in tqdm(range(20)):
#     test.equilibrate()
#     for i in range(10):
#         test._run(steps=200)
#         diff_mtx = test._pairwise_diff_vector_matrix()
#         dist_mtx = np.linalg.norm(diff_mtx, axis=-1)
#         triu_idx = np.triu_indices_from(dist_mtx, k=1)
#         distances = dist_mtx[triu_idx]
#         counts, bins, hist = plt.hist(distances, bins=400, range=(0,test.boxsize))
#         set_counts.append(counts)
#     test.reset()

# we have a total list of 10 arrays containing the binned counts of pairwise distances. 
# we can average these arrays binwise by taking sum and dividing by 10. 
# avg_counts = np.mean(np.array(set_counts), axis=0)

# def pair_correlation_func(n_r, r):
#     volume = test.boxsize**test.dim
#     g = ((2*volume) / (test.num_particles * (test.num_particles - 1))) * \
#         (n_r / (4 * np.pi * r**2 * (test.boxsize/400)))
#     return g

# plt.show() # all the trash histograms out of the way

# fig, (ax1, ax2) = plt.subplots(2,1)

# ax1.set_title('pairwise distances')

test.run_ensemble()
radii, pcf = test.pair_corr_function()
test.reset()

# radii = np.linspace(0.001, test.boxsize / 2, 400)
# pcf = pair_correlation_func(avg_counts, radii)

plt.plot(radii, pcf)
plt.title('pair correlation function')
plt.tight_layout()
plt.show()