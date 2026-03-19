# cpa_group5

Molecular dynamics simulations of Lennard-Jones particles.

## Installation
git clone https://github.com/phstoot/Project-1-atoms
pip install -r requirements.txt

## Main Idea
Initialize a system by creation a simulation object:
sim = Simulation(density: float = 0.8, 
        temp: float = 1,  
        num_particles: int = 108,
        dim: int = 3,
        timestep_h: float = 0.001,
        units: str = "natural",
        optimized: bool = False,
        rcutoff: float = 2.7,
        numba=False)

The input variables for Simulation(args) are the main arguments to play around with, providing a rich variety of options.
density: Density of the system, used when calculating the system size for some particle number.
temp: Temperature. During .equilibrate(), the exact temperature will be found.
num_particles: Number of particles
dim: Dimension. 2D or 3D, although 2D has not been implemented yet.
timestep_h: Time window between consecutive integration steps.
untis: natural or SI, although SI has not been implemented yet.
optimized: if True, will run optimized version of code (however, only at system sizes bigger than 3* r_cutoff)
rcutoff: Cutoff radius, beyond which interactions will not be considered.
numba: If True, combined with optimized, will use a numba optimized version of the code.

Note: The parameters can be manually edited in between runs, using self.param.

## Structure
- simulation.py: Establishes simulation class and all functions related to it
- utils.py: Establishes external functions, e.g. necessary for the numba optimized algorithm.

## Usage
Demo scripts (see below):
python demo_*.py

A simple algorithm to become familiar:
from simulation import Simulation
sim = Simulation(density=0.8, temp=1.0)
sim.equilibrate()
sim.run()

## After run
If a script is run multiple times outside of the terminal, it is best practice to reset the simulation object over
sim.reset()

## Demo scripts
- demo_results.py — reproduces all report results
- demo_animation.py — live animation feature
- demo_bonus.py — optimized simulation for large N