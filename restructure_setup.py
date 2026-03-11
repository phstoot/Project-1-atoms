import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from functions import interaction_force, lennard_jones_potential


class PositiveInteger:
    """A descriptor class to validate if a value is an integer and positive."""
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{self._name} must be an integer, got {value}")
            warnings.warn(f"{self._name} was passed as float ({value}), converting to int")
            value = int(value)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{self._name} must be a positive integer")
        instance.__dict__[self._name] = value

class PositiveFloat:
    """A descriptor class to validate if a value is a float and positive."""
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float) or value <= 0:
            raise ValueError(f"{self._name} must be a positive float")
        instance.__dict__[self._name] = value


class Simulation:
    """A simulation class used to build and run an Argon molecular dynamics simulation. 

    Parameters
    ----------
    density : float
        _description_
    num_particles : int, optional
        _description_, by default 108
    boxsize : float, optional
        _description_, by default 10
    dim : int, optional
        _description_, by default 2
    timestep_h : float, optional
        _description_, by default 0.001
    units : str, optional
        _description_, by default 'natural'

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    # Descriptors for repeated validation rules
    density      = PositiveFloat()
    temp         = PositiveFloat()
    boxsize      = PositiveFloat()
    timestep_h   = PositiveFloat()
    num_particles = PositiveInteger()

    def __init__(
            self, 
            density : float     = 1, #FOR NOW # prevent negative values (with @property?)
            temp : float        = 1, #FOR NOW
            num_particles : int = 10, 
            boxsize : float     = 10, 
            dim : int           = 2, 
            timestep_h : float  = 0.001, 
            units : str         = 'natural'

        ):
        """_summary_

        Parameters
        ----------
        density : float
            _description_
        num_particles : int, optional
            _description_, by default 108
        boxsize : float, optional
            _description_, by default 10
        dim : int, optional
            _description_, by default 2
        timestep_h : float, optional
            _description_, by default 0.001
        units : str, optional
            _description_, by default 'natural'
        """
        
        self.density         = density
        self.temp            = temp
        self.num_particles   = num_particles
        self.boxsize         = boxsize
        self.dim             = dim            # goes through @property setter
        self.timestep_h      = timestep_h
        self.units           = units          # goes through @property setter
        
        self.positions       = self._init_positions()
        self.velocities      = self._init_velocities()
        self.forces          = self._net_forces()
        
        self.positions_hist  = []
        self.velocities_hist = []
        self.forces_hist     = []
        self.e_kin_hist      = []
        self.e_pot_hist      = []
        self.e_tot_hist      = []



        pass

    @property
    def dim(self):
        return self._dim
    
    @dim.setter
    def dim(self, value):
        if isinstance(value, float):
            value = int(value)
        if not value in (2, 3):
            raise ValueError(f"dim must be 2 or 3 (integer), got {value!r}")
        self._dim = value

    @property
    def units(self):
        return self._units
    
    @units.setter
    def units(self, value):
        if not isinstance(value, str) or not value in ('natural', 'SI'):
            raise ValueError(f"units must be 'natural' or 'SI', got {value!r}")
        self._units = value

    def __repr__(self):
        return (f"Simulation(density={self.density}, temp={self.temp}, "
                f"num_particles={self.num_particles}, boxsize={self.boxsize}, "
                f"dim={self.dim}, timestep_h={self.timestep_h}, units={self.units})")

    def _init_positions(self):
        """Private method to initialize positions. FOR NOW: get working with usual random init.
        """
        return np.random.uniform(0, self.boxsize, size=(self.num_particles, self.dim))
    
    def _init_velocities(self):
        """Private method to initialize velocities. FOR NOW: get working with usual random init.
        """
        return np.random.uniform(-4*self.boxsize, 4*self.boxsize, size=(self.num_particles, self.dim))

    def _pairwise_diff_vector_matrix(self):
        """Generate a pairwise vector matrix with e_ij the vector between particle i and j.
        The minimal image convention is implemented.
        """
        diff = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :] # shape (N, N, dim)
        diff = (diff + 0.5*self.boxsize) % self.boxsize - 0.5*self.boxsize # minimal image convention
        return diff
    
    def _net_forces(self):
        """The pairwise vector matrix is used to find the interaction 
        force between each particle pair, then all forces are summed 
        over an axis to find the net force on each particle. Returns final array to 
        enable storage of multiple force arrays in one simulation step.
        """
        diff = self._pairwise_diff_vector_matrix()
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf) # mask diagonals to prevent division by zero
        F_mag = -interaction_force(dist)
        F_matrix = F_mag[:, :, np.newaxis] * diff
        return F_matrix.sum(axis=1)
    
    def _update_positions(self):
        """Private method: use Verlet's algorithm to update positions, with box constraints applied. Part of general step in the simulation.
        """
        self.positions += self.timestep_h * self.velocities + 0.5 * (self.timestep_h**2) * self.forces
        self.positions %= self.boxsize

    def _update_velocities(self, new_forces):
        """Private method: use Verlet's algorithm to update velocities. Part of general step in the simulation.
        """
        self.velocities += 0.5 * self.timestep_h * (self.forces + new_forces)
    
    def _kinetic_energy(self):
        return 0.5 * np.sum(self.velocities**2) # sum squared components identical to summing squared vector norms
    
    def _potential_energy(self):
        diff = self._pairwise_diff_vector_matrix() # OPTIMIZE: the matrix calculation (O(N^2)) is done both here and in force method
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf) # mask diagonals to prevent division by zero
        return 0.5 * np.sum(lennard_jones_potential(dist))

    def _step(self):
        """Private method: Advance the system by one step with Verlet's Algorithm
        """
        self._update_positions()            # with self.forces from initialization or previous step
        new_forces = self._net_forces()
        self._update_velocities(new_forces) # uses both F(t) and F(t+h)
        self.forces = new_forces            # roll forward
    
    def run(self, steps: int=1000, sample_interval: int | None = None):
        """Run the simulation. implement sample interval for long simulations to keep history size manageable
        """
        if sample_interval is None:
            sample_interval = max(1, steps // 1000)
        
        for step in range(steps):
            if step % sample_interval == 0:
                self.positions_hist.append(self.positions.copy())
                self.velocities_hist.append(self.velocities.copy())
                self.forces_hist.append(self.forces.copy())
                self.e_kin_hist.append(self._kinetic_energy())
                self.e_pot_hist.append(self._potential_energy())
            self._step()
            

# test = Simulation()



    # def run_live(self): # unfinished
    #     fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6,8), height_ratios=[6,3])
    #     fig.suptitle('Live animation window')
    #     ax = fig.add_subplot(2,1,1, projection='3d')
    #     ax.set_xlim(0, self.boxsize)
    #     ax.set_ylim(0, self.boxsize)
    #     ax.set_zlim(0, self.boxsize)
    #     ax.set_aspect('equal')
    #     ax.grid(False)
    #     scat = ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2])

    #     ax2 = fig.add_subplot(2,1,2)
    #     (plot_kin,) = ax2.plot([], [], label="E_kin")
    #     (plot_pot,) = ax2.plot([], [], label="E_pot")
    #     # rolling window size
    #     repeat_length = 500
    #     ax2.set_xlim([0, repeat_length])
    #     ax2.set_ylim([(0 - 0.1 * kin_0) / N, (kin_0 + 0.1 * kin_0) / N])
    #     # ax2.set_yscale("symlog", linthresh=1e-2)
    #     ax2.legend(loc=7)

    #     def update(self, frame):
    #         self._step() # advance system by one step
    #         scat._offsets3d = 

    
    # def equilibrate(self, density, temp):
    #     # see lecture 4 and coding guidelines
    #     pass

    # def simulate(self, algorithm, time, *, live_animation=True, store_arrays=True):
    #     # see lecture 4
    #     # time how long to simulate?
    #     # algorithm: euler or verlet

    #     #i'd say: based on chosen algorithm, import correct function
    #     # from other file
    #     pass

    # def quickshow(self):
    #     # just an idea
    #     pass

    # def deconstruct(self):
    #     pass

    # def save_animation(self):
    #     # and specify in what form
    #     pass


