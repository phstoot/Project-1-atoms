import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import matplotlib.animation as animation
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
    optimized : bool, optional
        _description_, by default False
    cutoff: float, optional
        _description_, by default 2.71

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
            density : float     = 0.8, #FOR NOW 
            temp : float        = 1, #FOR NOW
            num_particles : int = 108, 
            dim : int           = 3, 
            timestep_h : float  = 0.001, 
            units : str         = 'natural',
            optimized:  bool = False,
            rcutoff: float = 2.43

        ):
        """_summary_

        Parameters
        ----------
        density : float
            _description_
        num_particles : int, optional
            _description_, by default 108
        dim : int, optional
            _description_, by default 2
        timestep_h : float, optional
            _description_, by default 0.001
        units : str, optional
            _description_, by default 'natural'
        optimized : bool, optional
            _description_, by default False
            cutoff: float, optional
        _description_, by default 2.43
        """
        self.density         = density
        self.temp            = temp
        self.num_particles   = num_particles
        self.dim             = dim            # goes through @property setter
        self.timestep_h      = timestep_h
        self.units           = units          # goes through @property setter
        self.optimized       = optimized
        self.rcutoff         = rcutoff # for linked-cell algorithm
        
        self.boxsize         = (self.num_particles / self.density) ** (1/self.dim) # derived from density 
        self.positions       = self._init_positions()
        self.velocities      = self._init_velocities()
        self.forces          = self._net_forces()
        
        self.positions_hist  = []
        self.velocities_hist = []
        self.forces_hist     = []
        self.e_kin_hist      = []
        self.e_pot_hist      = []
        self.e_tot_hist      = [] # STILL UNUSED
        self.timestep        = 0  # STILL UNUSED, but could be useful for tracking time in live animation or long simulations
        self._status         = 'initialized'

    @property
    def dim(self):
        return self._dim
    
    @dim.setter
    def dim(self, value):
        if isinstance(value, float):
            value = int(value)
        if not value in (2, 3):
            raise ValueError(f"dim must be 2 or 3 (integer), got {value!r}")
        elif value == 2:
            raise ValueError("The code does not work yet for 2D (position grid init)")
        self._dim = value

    @property
    def units(self):
        return self._units
    
    @units.setter
    def units(self, value):
        if not isinstance(value, str) or not value in ('natural', 'SI'):
            raise ValueError(f"units must be 'natural' or 'SI', got {value!r}")
        self._units = value

    @property
    def status(self):
        return self._status
    
    # @status.setter
    # def status(self, value):
    #     if not isinstance(value, str) or not value in ('initialized', 'equilibrated', 'completed'):
    #         raise ValueError(f"Simulation status is unknown. Call reset()")
    #     self._status = value

    def __repr__(self):
        return (f"Simulation(density={self.density:.2f}, input temp={self.temp:.2f}, "
                f"num_particles={self.num_particles}, boxsize={self.boxsize:.2f}, "
                f"dim={self.dim}, timestep_h={self.timestep_h}, units={self.units} "
                f"| status: {self._status})")

    def _init_positions(self):
        """Private method to initialize positions based on the face-centered cubic (FCC) lattice, the assumed 
        starting configuration for Argon.
        """
        basis = np.array([[0,   0,   0. ],
                          [0.5, 0.5, 0. ],
                          [0.5, 0,   0.5],
                          [0,   0.5, 0.5]])
        n = int(np.ceil((self.num_particles / 4) ** (1/3))) # round up, not down
        if abs(4 * n**3 - self.num_particles) > 0:
            warnings.warn(f"num_particles={self.num_particles} is not a perfect FCC number. "
                          f"Nearest valid values are {4*(n-1)**3} or {4*(n)**3}.")
        cell_size = self.boxsize / n

        positions = []
        for ix in range(n):
            for iy in range(n):
                for iz in range(n):
                    offset = np.array([ix, iy, iz]) * cell_size
                    for b in basis:
                        positions.append(offset + b * cell_size)
        return np.array(positions[:self.num_particles])
    
    def _init_velocities(self):
        """Private method to initialize velocities. We draw samples from a gaussian distribution centred
        at zero and with variance = temperature (natural units).
        """
        vel = np.random.normal(0, np.sqrt(self.temp), (self.num_particles, self.dim))
        vel -= vel.mean(axis=0) # subtract mean velocity drift to prevent net momentum
        return vel

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
    
    def _compute_cell_indices(self):
        """Private method to compute the cell index for each particle, used for cell list algorithm.
        """
        return np.floor(self.positions / self.rcutoff).astype(int)
    
    def _build_cell_list(self):
        """Private method to store particles in a cell list, used to identify which particles interact significantly. 
        Cell_list is updated differently dependent on the optimization algorithm.
        """
        self.cell_list = defaultdict(list)

        for i, cell in enumerate(self._compute_cell_indices()):
            key = tuple(cell)
            self.cell_list[key].append(i)
            
    def _net_forces_cell_list(self):
        """Private method: optimized force calculation algorithm; called cell list algorithm.
        Only forces between particles of adjacent cells are calculated, with the efficiency scaling with the system size - the algorithm is only used for systems of more than 2 cells per dimension.
        Implements periodic boundary condition, but not minimum image convention. 
        Uses matrix calculation instead of nested for-loops. 
        Returns final array to enable storage of multiple force arrays in one simulation step.
        """
        forces = np.zeros_like(self.positions)
        self._build_cell_list()

        for cell, particles in self.cell_list.items():
            particles = np.array(particles)

            adjacent_cells = [
                (dx, dy, dz)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                for dz in [-1, 0, 1]
            ]
            
            for shift in adjacent_cells:
                neighbor_cell = neighbor_cell = tuple(
                    (np.array(cell) + shift) % int(self.boxsize / self.rcutoff)
                ) # Implements periodic boundary conditions

                if neighbor_cell not in self.cell_list:
                    continue

                neighbors = np.array(self.cell_list[neighbor_cell])
                pos_i = self.positions[particles]     # shape (Ni, 3)
                pos_j = self.positions[neighbors]     # shape (Nj, 3)

                diff = pos_j[np.newaxis, :, :] - pos_i[:, np.newaxis, :]
                diff = (diff + 0.5*self.boxsize) % self.boxsize - 0.5*self.boxsize

                # Masking
                dist2 = np.sum(diff**2, axis=-1)
                mask = (dist2 < self.rcutoff**2) & (dist2 > 0)
                valid_diff = diff[mask]
                valid_dist = np.sqrt(dist2[mask])

                F_mag = -interaction_force(valid_dist)
                F_vec = (F_mag[:, np.newaxis] * valid_diff)

                # Scattering back to keep particle identity
                idx_i, idx_j = np.where(mask)
                np.add.at(forces, particles[idx_i], F_vec)
                np.add.at(forces, neighbors[idx_j], -F_vec)

        return forces
    
    def _update_positions(self, alg: str="verlet"):
        """Private method: use Verlet's algorithm to update positions, with box constraints applied. Part of general step in the simulation.
        """
        if alg == "verlet":
            self.positions += self.timestep_h * self.velocities + 0.5 * (self.timestep_h**2) * self.forces
            self.positions %= self.boxsize
        if alg == "euler": # for testing only, not recommended for actual simulation
            self.positions += self.timestep_h * self.velocities
            self.positions %= self.boxsize

    def _update_velocities(self, new_forces, alg: str="verlet"):
        """Private method: use Verlet's algorithm to update velocities. Part of general step in the simulation.
        """
        if alg == "verlet":
            self.velocities += 0.5 * self.timestep_h * (self.forces + new_forces)
        if alg == "euler":
            self.velocities += self.forces * self.timestep_h
    
    def _kinetic_energy(self) -> float:
        return 0.5 * np.sum( self.velocities**2 ) # sum squared components identical to summing squared vector norms
    
    def _potential_energy(self) -> float:
        diff = self._pairwise_diff_vector_matrix() # OPTIMIZE: the matrix calculation (O(N^2)) is done both here and in force method
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf) # mask diagonals to prevent division by zero
        return 0.5 * np.sum(lennard_jones_potential(dist))

    def _step(self, alg: str="verlet"):
        """Private method: Advance the system by one step with Verlet's Algorithm. Force calculation depends on box size and if optimization is turned on.
        """
        if alg == "verlet":
            self._update_positions(alg="verlet")            # with self.forces from initialization or previous step
            if self.optimized and int(self.boxsize / self.rcutoff) > 2: # cell list algorithm only makes sense if we have at least 3 cells per dimension
                new_forces = self._net_forces_cell_list()   # optimized force calculation
            else:
                new_forces = self._net_forces()
            self._update_velocities(new_forces, alg= "verlet") # uses both F(t) and F(t+h)
            self.forces = new_forces            # roll forward
        
        if alg == "euler":
            self._update_positions(alg="euler")
            self._update_velocities(None, alg="euler")
            if self.optimized and int(self.boxsize / self.rcutoff) > 2: # cell list algorithm only makes sense if we have at least 3 cells per dimension
                new_forces = self._net_forces_cell_list()   # optimized force calculation
            else:
                new_forces = self._net_forces()
    
    def _run(self, steps: int=1000):
        """Private method for running the simulation without storing history and without status checks, 
        used in equilibrate().
        """
        for step in range(steps):
            self._step()

    def run(self, steps: int=1000, sample_interval: int | None = None, store: bool = True, verbose: bool = True, alg: str="verlet"):
        """Run the simulation. Implement sample interval for long simulations to keep history 
        size manageable. Store = False will not store any history of simulation data, only the final
        state can be accessed in attributes.
        """
        if self._status == "initialized":
            raise RuntimeError("System has not been equilibrated. Call equilibrate() first.")
        if self._status == "completed":
            raise RuntimeError("run() already called. Call reset() to start fresh.")
        
        if sample_interval is None:
            sample_interval = max(1, steps // 1000)
        
        for step in tqdm(range(steps)):
            if store and step % sample_interval == 0:
                self.positions_hist.append(self.positions.copy())
                self.velocities_hist.append(self.velocities.copy())
                self.forces_hist.append(self.forces.copy())
                self.e_kin_hist.append(self._kinetic_energy())
                self.e_pot_hist.append(self._potential_energy())
            self._step(alg=alg)
        if verbose:
            print(f'Run completed ({steps} steps)')
        self._status = 'completed'
        
    def equilibrate(self, steps_between: int = 200, max_rescalings: int = 100):
        """Equilibrate the system towards target temperature after initialisation.
        
        After initialisation of the Simulation instance, the first step is to 
        guide the system towards equilibrium by calling this method.
        """
        if self._status == "equilibrated":
            raise RuntimeError("System is already in equilibrium. Call run() to run simulation.")
        if self._status == "completed":
            raise RuntimeError("run() already called. Call reset() to start fresh.")
        
        temp_measured = 0.0                 # prevent unbound variable
        stop = 0                            # prevent unbound variable
        self._run(steps=steps_between)    # initial disordering of lattice
        for _ in range(max_rescalings):
            temp_measured = (2 * self._kinetic_energy()) / (self.dim * (self.num_particles - 1))
            ratio = abs(temp_measured - self.temp) / self.temp
            if ratio < 0.01:
                stop = _ + 1
                self._status = 'equilibrated'
                print(f'Equilibrated at T = {temp_measured:.4f} (original input: T = {self.temp}), within {stop} rescalings')
                break
            lambda_ = np.sqrt(self.temp / temp_measured)
            self.velocities *= lambda_
            self._run(steps=steps_between)
        else:
            warnings.warn(f'Equilibration did not converge within {max_rescalings} rescalings.')
        
    def measure_temp(self):
        return (2 * self._kinetic_energy()) / (self.dim * (self.num_particles - 1))
    
    def print_status(self):
        print(f'Current status: {self._status}')

    def reset(self):
        print('Resetting the simulation... Input parameters like density, temperature, ... remain unchanged')
        self.positions       = self._init_positions()
        self.velocities      = self._init_velocities()
        self.forces          = self._net_forces()
        self.positions_hist  = []
        self.velocities_hist = []
        self.forces_hist     = []
        self.e_kin_hist      = []
        self.e_pot_hist      = []
        self.e_tot_hist      = []
        self.timestep        = 0
        self._status         = 'initialized'

    def _update_animation(self, frame, alg: str= "verlet", store: bool = True): # I think "store" needs to be turned on always - for the animation?
        """This private method is essentially a copy of the run method with storing history,
        with some additional code for the live animation. 
        """
        if store:
            self.positions_hist.append(self.positions.copy())
            self.velocities_hist.append(self.velocities.copy())
            self.forces_hist.append(self.forces.copy())
            self.e_kin_hist.append(self._kinetic_energy())
            self.e_pot_hist.append(self._potential_energy())
        self._step(alg=alg)
        
        self.scat._offsets3d = (self.positions[:,0], self.positions[:,1], self.positions[:,2])
        
        start = max(0, len(self.e_kin_hist) - self.steps_window)
        
        self.plot_kin.set_xdata(np.arange(len(self.e_kin_hist)))
        self.plot_kin.set_ydata(np.array(self.e_kin_hist) / self.num_particles)
        self.plot_pot.set_xdata(np.arange(len(self.e_pot_hist)))
        self.plot_pot.set_ydata(np.array(self.e_pot_hist) / self.num_particles)
        self.plot_tot.set_xdata(np.arange(len(self.e_kin_hist)))
        self.plot_tot.set_ydata((np.array(self.e_kin_hist) + np.array(self.e_pot_hist)) / self.num_particles)

        
        self.ax2.set_xlim(start, start + self.steps_window)
                           
        return self.scat, self.plot_kin, self.plot_pot
    
    
    def run_live(self, steps: int=1000):
        """Runs the simulation continouusly. Plots the energy evolution per particle in real-time.
        """
        if self._status == "initialized":
            raise RuntimeError("System has not been equilibrated. Call equilibrate() first.")
        if self._status == "completed":
            raise RuntimeError("run() or run_live() already called. Call reset() to start fresh.")
        
        self.steps_window = steps
        
        fig = plt.figure(figsize=(6, 8))
        fig.suptitle('Live animation window')
        
        #3D axis
        self.ax = fig.add_subplot(2,1,1, projection='3d')
        self.ax.set_xlim(0, self.boxsize)
        self.ax.set_ylim(0, self.boxsize)
        self.ax.set_zlim(0, self.boxsize) # Only for 3D, not for 2D
        self.ax.set_aspect("equal")
        self.ax.grid(False)
        
        self.scat = self.ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2])
        
        # Second Axes: energy evolution
        self.ax2 = fig.add_subplot(2, 1, 2)
        (self.plot_kin,) = self.ax2.plot([], [], label=r"E$_{kin}$")
        (self.plot_pot,) = self.ax2.plot([], [], label=r"E$_{pot}$")
        (self.plot_tot,) = self.ax2.plot([], [], label=r"E$_{total}$")
        
        # for i in range(self.timestep): # Probably not necessary, but in case run_live is called after some steps have already been taken
        #     self.e_kin_hist.append(self._kinetic_energy())
        #     self.e_pot_hist.append(self._potential_energy())

        # rolling window size
        mid = ((self._potential_energy() + self._kinetic_energy()) / 2) / self.num_particles
        self.ax2.set_xlim(left=(max(0, len(self.e_kin_hist) - steps)), right=max(steps, len(self.e_kin_hist)))
        self.ax2.set_ylim(bottom=(mid - 2.2*mid), top=(mid + 2.2*mid))
        self.ax2.legend(loc=7)
        
        self.ani = animation.FuncAnimation(
            fig,
            self._update_animation,
            frames=steps,
            interval=20,
            blit=False,
            repeat=False,
        )

        plt.show()
        self._status = 'completed'
    

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


