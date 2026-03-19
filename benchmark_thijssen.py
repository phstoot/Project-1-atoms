"""This script reproduces the simulations shown in Table 8.1 from Thijssen (2007). 
It is intended as a benchmark test of our simulation code.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from cpa_group5.simulation import Simulation
from cpa_group5.utils import spacer, section

if __name__ == '__main__':
    bench1 = Simulation(density=0.88, temp=1)
    bench2 = Simulation(density=0.8, temp=1)
    bench3 = Simulation(density=0.7, temp=1)

    # obtain measured T and U
    bench1.reset()
    bench1.equilibrate()
    bench1.run(steps=2000)
    bench1_temp = bench1.measure_temp()
    bench1_U = np.mean(np.array(bench1.e_pot_hist) / bench1.num_particles)

    bench2.reset()
    bench2.equilibrate()
    bench2.run(steps=2000)
    bench2_temp = bench2.measure_temp()
    bench2_U = np.mean(np.array(bench2.e_pot_hist) / bench2.num_particles)

    bench3.reset()
    bench3.equilibrate()
    bench3.run(steps=2000)
    bench3_temp = bench3.measure_temp()
    bench3_U = np.mean(np.array(bench3.e_pot_hist) / bench3.num_particles)

    # obtain pressures
    bench1.run_ensemble(n_resets=5, steps=1000, sample_interval=100, verbose=False)
    bench2.run_ensemble(n_resets=5, steps=1000, sample_interval=100, verbose=False)
    bench3.run_ensemble(n_resets=5, steps=1000, sample_interval=100, verbose=False)

    bench1_r, bench1_pcf = bench1.measure_pair_corr_function()
    bench1_pressure = bench1.measure_pressure()
    bench1_pressure_mean = np.mean(bench1_pressure)
    bench1_pressure_error = np.std(bench1_pressure)

    bench2_r, bench2_pcf = bench2.measure_pair_corr_function()
    bench2_pressure = bench2.measure_pressure()
    bench2_pressure_mean = np.mean(bench2_pressure)
    bench2_pressure_error = np.std(bench2_pressure)

    bench3_r, bench3_pcf = bench3.measure_pair_corr_function()
    bench3_pressure = bench3.measure_pressure()
    bench3_pressure_mean = np.mean(bench3_pressure)
    bench3_pressure_error = np.std(bench3_pressure)

    # obtain compressibility factor betaP/rho
    def comp_factor(P, rho, T):
        factor = P / (rho * T)
        return factor

    factor1 = comp_factor(bench1_pressure_mean, bench1.density, bench1_temp)
    factor2 = comp_factor(bench2_pressure_mean, bench2.density, bench2_temp)
    factor3 = comp_factor(bench3_pressure_mean, bench3.density, bench3_temp)

    # reproduce table 8.1 from Thijssen 2007
    print(
        f"{'density':<10} {'T_0':<10} {'T_measured':<10} {'P_mean':<10} {'P_std':<10} {'comp_factor':<10} {'U':<10}\n"
    )
    print("-" * 65 + "\n")
    print(
        f"{bench1.density:<10.3f} {bench1.temp:<10.3f} {bench1_temp:<10.3f} {bench1_pressure_mean:<10.4f} {bench1_pressure_error:<10.4f} {factor1:<10.4f} {bench1_U:<10.4f}\n"
    )
    print(
        f"{bench2.density:<10.3f} {bench2.temp:<10.3f} {bench2_temp:<10.3f} {bench2_pressure_mean:<10.4f} {bench2_pressure_error:<10.4f} {factor2:<10.4f} {bench2_U:<10.4f}\n"
    )
    print(
        f"{bench3.density:<10.3f} {bench3.temp:<10.3f} {bench3_temp:<10.3f} {bench3_pressure_mean:<10.4f} {bench3_pressure_error:<10.4f} {factor3:<10.4f} {bench3_U:<10.4f}\n"
    )