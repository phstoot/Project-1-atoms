from simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from functions import lennard_jones_potential, interaction_force, min_vector

# radii_small = np.linspace(10**-3,1,1000, dtype=float)
# radii_large = np.linspace(1,5,1000, dtype=float)

# potentials_small = lennard_jones_potential(radii_small)
# potentials_large = lennard_jones_potential(radii_large)
# force_small = interaction_force(radii_small)
# force_large = interaction_force(radii_large)

# fig, (ax1, ax2) = plt.subplots(2,1)
# ax1.plot(radii_small, potentials_small)
# ax2.plot(radii_large, potentials_large)
# ax1.set_yscale("log")
# plt.show()

if __name__ == '__main__':
    test_simulation = Simulation()
    print(f"Simulation instance created:")
    print(test_simulation)
    test_simulation.equilibrate()
    print(f"Status: {test_simulation.status}")
    print(f"Temp after equilibrate: {test_simulation.measure_temp():.4f}")
    test_simulation.run()
    print(f"Status: {test_simulation.status}")
    print(f"Temp after run: {test_simulation.measure_temp():.4f}")
    print('Plotting...')

# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(test.positions[:,0], test.positions[:,1], test.positions[:,2])
# ax.set_aspect('equal')
# ax.set_title('initial positions')
# plt.show()

# fig = plt.figure(figsize=(6,6))
# plt.hist(test.velocities.flatten(), bins=20)
# plt.title('initial velocity distribution')
# plt.show()

    fig2 = plt.figure(figsize=(12, 4))
    e_tot = [ek + ep for ek, ep in zip(test_simulation.e_kin_hist, test_simulation.e_pot_hist)]
    plt.plot(e_tot, label='total')
    plt.plot(test_simulation.e_kin_hist, label='kin')
    plt.plot(test_simulation.e_pot_hist, label ='pot')
    plt.title('energy evolution')
    plt.legend()
    plt.show()

# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(test.positions[:,0], test.positions[:,1], test.positions[:,2])
# ax.set_aspect('equal')
# ax.set_title('end positions')
# plt.show()

