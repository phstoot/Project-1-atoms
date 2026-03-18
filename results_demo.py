from simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from functions import lennard_jones_potential, interaction_force, min_vector

import matplotlib as mpl
from cycler import cycler

def collision_demo(alg):
    """demo to reproduce energy evolution plots

    Parameters
    ----------
    alg : str
        'euler' or 'verlet'
    """
    collision = Simulation(num_particles=2, dim=3, optimized=False)
    collision.positions = np.array([[-4.9, 3, 3], [4.9, 3, 3]], dtype=float)
    collision.velocities = np.array([[10, 0, 0], [-10, 0, 0]], dtype=float)
    collision.boxsize = 6
    collision._status = 'equilibrated'
    collision.run(steps=300, alg=alg)

    fig = plt.figure(figsize=(6,3))
    plt.plot(collision.e_kin_hist, label=r"E$_{kin}$")
    plt.plot(collision.e_pot_hist, label=r"E$_{pot}$")
    plt.plot((np.array(collision.e_kin_hist) + np.array(collision.e_pot_hist)), label=r"E$_{total}$")
    plt.xlabel(r'$t$ (steps)')
    plt.ylabel(r'$E$')
    plt.xlim(105,205)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'energies_{alg}.pdf')



if __name__ == '__main__':
    mpl.rcParams.update({'axes.labelsize': 13,
              'axes.prop_cycle': cycler('color', 'brcmyk'),
              'axes.titleweight': 'heavy',
              'axes.titlesize': 15,
              'figure.figsize': (6, 6),
              'font.family': ['serif'],
              'legend.fancybox':  False,
              'legend.edgecolor':     'black',
              'mathtext.fontset': 'dejavuserif',
              'patch.force_edgecolor': True,
              'xtick.direction': 'in',
              'xtick.top': True,
              'xtick.minor.visible': True,
              'xtick.major.size':    10, 
              'xtick.minor.size':    3, 
              'ytick.direction': 'in',
              'ytick.right': True,
              'ytick.minor.visible': True,
              'ytick.major.size':    10, 
              'ytick.minor.size':    3, })

    

    collision_demo('euler')
    collision_demo('verlet')


    gas = Simulation(density=0.3, temp=3) 
    liquid = Simulation(density=0.8, temp=1, num_particles=256)
    solid = Simulation(density=1.2, temp=0.5, num_particles=256)
    print('Three simulation instances were initialized corresponding to Argon gas, liquid and solid.')
    gas.run_ensemble(n_resets=50, steps=1000, sample_interval=50, verbose=False)
    liquid.run_ensemble(n_resets=40, steps=1000, sample_interval=100, verbose=False)
    solid.run_ensemble(n_resets=15, steps=2000, sample_interval=200, verbose=False)

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

    fig = plt.figure(figsize=(5,4))
    plt.plot(solid_r, solid_pcf, label='solid', c='c')
    plt.plot(liquid_r, liquid_pcf, label='liquid')
    plt.plot(gas_r, gas_pcf, label='gas')
    plt.legend()
    plt.xlabel(r'r / $\sigma$')
    plt.ylabel('g(r)')
    plt.xlim(0,3)
    plt.axhline(1, linewidth=0.5, color='grey', linestyle='--')
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5)) # type: ignore
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1)) # type: ignore
    # plt.show()
    plt.savefig('pcf.pdf')

    with open("pressures.txt", "w") as f:
        f.write(f"{'state':<10} {'density':<10} {'temperature':<15} {'pressure_mean':<15} {'pressure_std':<15}\n")
        f.write("-" * 65 + "\n")
        f.write(f"gas        {gas.density:<10.3f} {gas.temp:<15.3f} {gas_pressure_mean:<15.4f} {gas_pressure_error:<15.4f}\n")
        f.write(f"liquid     {liquid.density:<10.3f} {liquid.temp:<15.3f} {liquid_pressure_mean:<15.4f} {liquid_pressure_error:<15.4f}\n")
        f.write(f"solid      {solid.density:<10.3f} {solid.temp:<15.3f} {solid_pressure_mean:<15.4f} {solid_pressure_error:<15.4f}\n")

    