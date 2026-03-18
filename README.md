# cpa_group5

Molecular dynamics simulations of Lennard-Jones particles.

## Installation
clone the repo from github, make sure to check requirements.txt

## Usage
from simulation import Simulation
sim = Simulation(density=0.8, temp=1.0)
sim.equilibrate()
sim.run()

## Demo scripts
- demo_results.py — reproduces all report results
- demo_animation.py — live animation feature
- demo_bonus.py — optimized simulation for large N