from simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from functions import lennard_jones_potential, interaction_force, min_vector

import matplotlib as mpl
from cycler import cycler





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


    gas = Simulation(density=0.3, temp=3) 
    liquid = Simulation(density=0.8, temp=1)
    solid = Simulation(density=1.2, temp=0.5)
    print('Three simulation instances were initialized corresponding to Argon gas, liquid and solid.')
    gas.run_ensemble(n_resets=50, steps=1000, sample_interval=50, verbose=False)
    liquid.run_ensemble(n_resets=40, steps=1000, sample_interval=100, verbose=False)
    solid.run_ensemble(n_resets=15, steps=2000, sample_interval=200, verbose=False)

    gas_r, gas_pcf = gas.measure_pair_corr_function()
    gas_pressure = gas.measure_pressure()

    liquid_r, liquid_pcf = liquid.measure_pair_corr_function()
    liquid_pressure = liquid.measure_pressure()

    solid_r, solid_pcf = solid.measure_pair_corr_function()
    solid_pressure = solid.measure_pressure()

    fig = plt.figure(figsize=(5,4))
    plt.plot(gas_r, gas_pcf, label='gas')
    plt.plot(liquid_r, liquid_pcf, label='liquid')
    plt.plot(solid_r, solid_pcf, label='solid')
    plt.legend()
    plt.xlabel(r'r / $\sigma$')
    plt.ylabel('g(r)')
    plt.xlim(0,2.2)
    plt.axhline(1, linewidth=0.5, color='grey', linestyle='--')
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.show()

    print(f'gas pressure: {gas_pressure:.2f}')
    print(f'liquid pressure: {liquid_pressure:.2f}')
    print(f'solid pressure: {solid_pressure:.2f}')
    