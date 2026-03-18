"""
Script to demonstrate live animation feature

Usage:
    python demo_animation.py

Output:
    Matplotlib animation window displaying the 3D system box and the live energy evolution.
"""

from simulation import Simulation
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
from utils import spacer, section
import matplotlib as mpl
from cycler import cycler

if __name__ == "__main__":
    section("overview")
    print("This script demonstrates the live animation of the module.")
    spacer()
    sleep(1)
    sim = Simulation(density=1.2, temp=0.5)
    print(f"Simulation instance created:")
    print(sim)
    spacer()
    sim.equilibrate()
    sim.run_live()
    sim.reset()