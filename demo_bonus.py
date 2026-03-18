"""
Script to demonstrate numba and cutoff_r optimization for large N simulations.

Usage:
    python demo_bonus.py

Output:
    - plot of FCC grid of 4000 particles
    - resulting energy evolution plot of simulation
"""

from cpa_group5.simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from tqdm import tqdm
from cpa_group5.utils import spacer, section

if __name__ == '__main__':
    section("overview")
    print("This script demonstrates the optimization of the module, running a simulation of 4000 particles (which happens to be a valid FCC multiple).")
    spacer()
    sleep(1)
    test_simulation = Simulation(num_particles=4000, temp=0.5, density=1.2, optimized=True, numba=True) # for testing purposes, we can set optimized and numba to False to check if the results are the same as with the original code. We can also set num_particles to a small number to check if the forces and energies are calculated correctly. We can also set temp to a high value to check if the system equilibrates correctly. We can also set density to a low value to check if the system behaves like an ideal gas. We can also set density to a high value to check if the system behaves like a solid. We can also set temp to a low value to check if the system behaves like a solid. We can also set temp to a high value and density to a low value to check if the system behaves like a gas. We can also set temp to a low value and density to a high value to check if the system behaves like a solid. We can also set temp to a high value and density to a high value to check if the system behaves like a liquid. We can also set temp to a low value and density to a low value to check if the system behaves like a gas.
    print(f"Simulation instance created:")
    print(test_simulation)
    spacer()
    
    print('Plotting...')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_simulation.positions[:,0], test_simulation.positions[:,1], test_simulation.positions[:,2])# type: ignore
    ax.set_aspect('equal')
    ax.set_title('initial positions')
    plt.show()

    print('Equilibrating...')
    test_simulation.equilibrate()
    print(f"Status: {test_simulation.status}")
    print(f"Temp after equilibrate: {test_simulation.measure_temp():.4f}")
    # test_simulation.run_live()
    test_simulation.run()
    print(f"Status: {test_simulation.status}")
    print(f"Temp after run: {test_simulation.measure_temp():.4f}")
    print('Plotting...')

    fig2 = plt.figure(figsize=(12, 4))
    e_tot = [ek + ep for ek, ep in zip(test_simulation.e_kin_hist, test_simulation.e_pot_hist)]
    plt.plot(e_tot, label=r'E$_{total}$')
    plt.plot(test_simulation.e_kin_hist, label=r'E$_{kin}$')
    plt.plot(test_simulation.e_pot_hist, label =r'E$_{pot}$')
    plt.title('Energy evolution')
    plt.xlabel(r"$t$ (steps)")
    plt.ylabel(r"$E$")
    plt.legend()
    plt.show()
    
    test_simulation.reset()

    print('exited')
