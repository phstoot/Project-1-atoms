"""
Script to reproduce all displayed results in the final report. Can take 10-20 minutes to run, due to simulation ensemble for sampling.
For a quick run, comment out the large run_ensemble() methods, and replace by smaller run_ensemble() methods below.

Usage:
    python demo_results.py

Output:
    - energy plots for two-particle collisions with euler and verlet algorithm
    - demonstration of a single simulation with FCC grid and randomly drawn velocities
    - pair correlation function plots (gas, liquid, solid)
    - pressures.txt with pressure values and error estimates
    - deviation of forces for linked cell and linked cell + numba optimization algorithms compared to normal algorithm
"""

from cpa_group5.simulation import Simulation
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
from cpa_group5.utils import spacer, section
import matplotlib as mpl
from cycler import cycler
import copy


def collision_demo(alg):
    """demo to reproduce energy evolution plots

    Parameters
    ----------
    alg : str
        'euler' or 'verlet'
    """
    print(f"Simulation object created:")
    collision = Simulation(num_particles=2, dim=3, optimized=False)
    print(collision)
    print(f"Algorithm: {alg}")
    spacer()
    collision.positions = np.array([[-4.9, 3, 3], [4.9, 3, 3]], dtype=float)
    collision.velocities = np.array([[10, 0, 0], [-10, 0, 0]], dtype=float)
    collision.boxsize = 6
    collision._status = "equilibrated"
    collision.run(steps=300, alg=alg)

    print("Plotting...")
    fig = plt.figure(figsize=(6, 3))
    plt.plot(collision.e_kin_hist, label=r"E$_{kin}$")
    plt.plot(collision.e_pot_hist, label=r"E$_{pot}$")
    plt.plot(
        (np.array(collision.e_kin_hist) + np.array(collision.e_pot_hist)),
        label=r"E$_{total}$",
    )
    plt.xlabel(r"$t$ (steps)")
    plt.ylabel(r"$E$")
    plt.xlim(105, 205)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"energies_{alg}.pdf")
    plt.show()
    spacer()

if __name__ == "__main__":
    section("overview")
    print("This script reproduces all results in the final report.")
    spacer()
    sleep(1)
    print(
        "Due to the simulation ensemble, it can take 10-20 minutes to run. (For a quick run, comment out the large run_ensemble() methods, and replace by the smaller run_ensemble() methods already supplied in script.)"
    )
    response = input("Continue? [y/n]: ").strip().lower()
    if response != "y":
        print("Exiting.")
        exit()
    sleep(1)
    mpl.rcParams.update(
        {
            "axes.labelsize": 13,
            "axes.prop_cycle": cycler("color", "brcmyk"),
            "axes.titleweight": "heavy",
            "axes.titlesize": 15,
            "figure.figsize": (6, 6),
            "font.family": ["serif"],
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "mathtext.fontset": "dejavuserif",
            "patch.force_edgecolor": True,
            "xtick.direction": "in",
            "xtick.top": True,
            "xtick.minor.visible": True,
            "xtick.major.size": 10,
            "xtick.minor.size": 3,
            "ytick.direction": "in",
            "ytick.right": True,
            "ytick.minor.visible": True,
            "ytick.major.size": 10,
            "ytick.minor.size": 3,
        }
    )

    section("collision: Euler vs Verlet")
    print(
        "In this section a two-particle collision is simulated using both Euler's and Verlet's algorithm for comparison."
    )
    spacer()
    collision_demo("euler")
    collision_demo("verlet")

    section("basic simulation")
    sim = Simulation()
    print(f"Simulation instance created:")
    print(sim)
    spacer()
    print("Plotting...")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sim.positions[:, 0], sim.positions[:, 1], sim.positions[:, 2])  # type: ignore
    ax.set_aspect("equal")
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.set_zticks([], minor=True)  # type: ignore
    plt.title(rf"$\rho$={sim.density}, $T$={sim.temp}", weight="normal", size=12)
    plt.tight_layout()
    plt.savefig("fcc_grid.pdf")
    plt.show()

    sim.equilibrate()
    sim.run()

    print("Plotting...")
    mpl.rcParams["axes.prop_cycle"] = mpl.rcParamsDefault["axes.prop_cycle"]
    fig = plt.figure(figsize=(6, 3))
    e_tot = [ek + ep for ek, ep in zip(sim.e_kin_hist, sim.e_pot_hist)]
    plt.plot(e_tot, label=r"E$_{total}$")
    plt.plot(sim.e_kin_hist, label=r"E$_{kin}$")
    plt.plot(sim.e_pot_hist, label=r"E$_{pot}$")
    mid = (sim._potential_energy() + sim._kinetic_energy()) / 2
    amp = sim._kinetic_energy() - mid
    plt.xlim(0, 1000)
    plt.ylim(bottom=(mid - 1.4 * amp), top=(mid + 1.4 * amp))
    plt.xlabel(r"$t$ (steps)")
    plt.ylabel(r"$E$")
    plt.tight_layout()
    plt.legend()
    # plt.savefig('energy_evolution.pdf')
    plt.show()

    section("simulation ensemble")
    gas = Simulation(density=0.3, temp=3)
    liquid = Simulation(density=0.8, temp=1, num_particles=256)
    solid = Simulation(density=1.2, temp=0.5, num_particles=256)
    print(
        "Three Simulation objects were initialized corresponding to Argon gas, liquid and solid:"
    )
    print(gas)
    print(liquid)
    print(solid)
    spacer()
    gas.run_ensemble(n_resets=50, steps=1000, sample_interval=50, verbose=False)
    liquid.run_ensemble(n_resets=40, steps=1000, sample_interval=100, verbose=False)
    solid.run_ensemble(n_resets=15, steps=2000, sample_interval=200, verbose=False)

    # For quick test runs:
    # gas.run_ensemble(n_resets=1, steps=1000, sample_interval=10, verbose=False)
    # liquid.run_ensemble(n_resets=1, steps=1000, sample_interval=10, verbose=False)
    # solid.run_ensemble(n_resets=1, steps=1000, sample_interval=10, verbose=False)

    section("pair correlation function")
    print("Plotting...")
    gas_r, gas_pcf = gas.measure_pair_corr_function()
    gas_pressure = gas.measure_pressure()
    gas_pressure_mean = np.mean(gas_pressure)
    gas_pressure_error = np.std(gas_pressure)

    liquid_r, liquid_pcf = liquid.measure_pair_corr_function()
    liquid_pressure = liquid.measure_pressure()
    liquid_pressure_mean = np.mean(liquid_pressure)
    liquid_pressure_error = np.std(liquid_pressure)

    solid_r, solid_pcf = solid.measure_pair_corr_function()
    solid_pressure = solid.measure_pressure()
    solid_pressure_mean = np.mean(solid_pressure)
    solid_pressure_error = np.std(solid_pressure)

    mpl.rcParams["axes.prop_cycle"] = cycler("color", "brcmyk")
    fig = plt.figure(figsize=(5, 4))
    plt.plot(solid_r, solid_pcf, label="solid", c="c")
    plt.plot(liquid_r, liquid_pcf, label="liquid")
    plt.plot(gas_r, gas_pcf, label="gas")
    plt.legend()
    plt.xlabel(r"r / $\sigma$")
    plt.ylabel("g(r)")
    plt.xlim(0, 3)
    plt.axhline(1, linewidth=0.5, color="grey", linestyle="--")
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))  # type: ignore
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))  # type: ignore
    plt.savefig("pcf.pdf")
    plt.show()

    section("pressure")
    print(
        f"{'state':<10} {'density':<10} {'temperature':<15} {'pressure_mean':<15} {'pressure_std':<15}\n"
    )
    print("-" * 65 + "\n")
    print(
        f"gas        {gas.density:<10.3f} {gas.temp:<15.3f} {gas_pressure_mean:<15.4f} {gas_pressure_error:<15.4f}\n"
    )
    print(
        f"liquid     {liquid.density:<10.3f} {liquid.temp:<15.3f} {liquid_pressure_mean:<15.4f} {liquid_pressure_error:<15.4f}\n"
    )
    print(
        f"solid      {solid.density:<10.3f} {solid.temp:<15.3f} {solid_pressure_mean:<15.4f} {solid_pressure_error:<15.4f}\n"
    )

    with open("pressures.txt", "w") as f:
        f.write(
            f"{'state':<10} {'density':<10} {'temperature':<15} {'pressure_mean':<15} {'pressure_std':<15}\n"
        )
        f.write("-" * 65 + "\n")
        f.write(
            f"gas        {gas.density:<10.3f} {gas.temp:<15.3f} {gas_pressure_mean:<15.4f} {gas_pressure_error:<15.4f}\n"
        )
        f.write(
            f"liquid     {liquid.density:<10.3f} {liquid.temp:<15.3f} {liquid_pressure_mean:<15.4f} {liquid_pressure_error:<15.4f}\n"
        )
        f.write(
            f"solid      {solid.density:<10.3f} {solid.temp:<15.3f} {solid_pressure_mean:<15.4f} {solid_pressure_error:<15.4f}\n"
        )
    
    section("optimization tests")
    num_particles = 864
    default_sim = Simulation(num_particles=num_particles, optimized=True, numba=True)
    print(
        f"A simulation with {num_particles} particles in a fluid state is initialized, such that the optimization algorithms can be tested for their effect on the data."
    )
    print(default_sim)
    print(
        "The system will be equilibrated, then copied for each type of algorithm."
    )
    spacer()

    default_sim.equilibrate()
    normal_sim = copy.deepcopy(default_sim)
    normal_sim.optimized = False
    linked_cell_sim = copy.deepcopy(default_sim)
    linked_cell_sim.numba = False
    linked_cell_numba_sim = copy.deepcopy(default_sim)

    print("Advancing over normal algorithm")
    normal_sim.run(steps=500)
    print("Advancing over linked_cell algorithm")
    linked_cell_sim.run(steps=500, )
    print("Advancing over linked_cell + numba algorithm")
    linked_cell_numba_sim.run(steps=500)

    deviation_linked_cell = [
        np.mean(np.linalg.norm(f_norm - f_lc, axis=1) / np.linalg.norm(f_norm, axis=1))
        for f_norm, f_lc in zip(
            normal_sim.forces_hist,
            linked_cell_sim.forces_hist
        )
    ]
    deviation_linked_cell_numba = [
        np.mean(np.linalg.norm(f_norm - f_nb, axis=1) / np.linalg.norm(f_norm, axis=1))
        for f_norm, f_nb in zip(
            normal_sim.forces_hist,
            linked_cell_numba_sim.forces_hist
        )
    ]

    fig = plt.figure(figsize=(6, 3))

    plt.plot(deviation_linked_cell, label=r"$\Delta F_{lc}$")
    plt.plot(deviation_linked_cell_numba, label=r"$\Delta F_{lcn}$")
    plt.xlim(0, 500)
    plt.xlabel(r"$t$ (steps)")
    plt.ylabel(r"$\Delta F$")
    plt.tight_layout()
    plt.legend()
    plt.savefig('force_deviations.pdf')
    plt.show()    