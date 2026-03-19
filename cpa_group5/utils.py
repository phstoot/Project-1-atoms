import numpy as np
import matplotlib.pyplot as plt
from random import randint
from numba import njit

## We could technically write all equations allowing for both natural and normal units (so using function(*,natural = True))

# rho = ...  # Particle density, not necessary for now
# T = ...  # temperature
# epsilon = 1.654 * 10 ** (-10)  # 119.8 #K (epsilon / k_boltzmann)
# sigma = 3.405 * 10 ** (-10)  # Angstrom
# mass = 6.6 * 10 ** (-26)  # Mass


def section(title: str):
    """print to console in nice format"""
    width = 60
    print("\n" + "=" * width)
    print(f"{title.upper():^{width}}")
    print("=" * width + "\n")


def spacer(n: int = 2):
    """print to console in nice format"""
    print("\n" * n, end="")


@njit
def interaction_force(r):
    """
    Calculates interaction force in natural units, based on the Lennard-Jones potential and already normalizing the connecting vector.

    Parameters
    ----------
    r : float
        Distance between two particles in natural units.

    Returns
    -------
    F : float
        Magnitude of Interaction Force in natural units.
    """

    F = 24 * (2 * r ** (-14) - r ** (-8))
    return F


def min_vector(part1, part2, L=10, dim=2):
    """
    Finds smallest vector connecting particle 1 to particle 2, in the smallest image convention.

    Parameters
    ----------
    part1 : arr
        Position of main particle in natural units.
    part2 : arr
        Position of interaction particle in natural units.
    L : float
        Size of the simulation box in natural units.
    dim : int
        Dimensionality of the system.

    Returns
    -------
    min_vec : arr
        Vector pointing to closest version of interaction particle in natural units.
    """
    vec = part2 - part1
    min_vec = np.mod(
        vec + np.full(dim, 0.5 * L, dtype=float), np.full(dim, L, dtype=float)
    ) - np.full(dim, 0.5 * L, dtype=float)
    return min_vec


## Energies
def Kinetic_Energies(vel):
    """
    Calculates the Kinetic Energy of each particle in natural units.

    Parameters
    ----------
    vel : arr
        Instantaneous velocity array in natural units.

    Returns
    -------
    E_kin : float
        Kinetic Energy array in natural units.
    """

    E_kin = 1 / 2 * vel * vel
    return E_kin


def lennard_jones_potential(r):
    """
    Calculates Potential Energy due to each particle interaction, in natural units.

    Parameters
    ----------
    r : float
        Distance between two particles in natural units.

    Returns
    -------
    U : float
        Potential Energy array in natural units.
    """

    U = 4 * (r ** (-12) - r ** (-6))
    return U


@njit
def compute_forces_numba(
    positions, boxsize, rcutoff, cell_particles, cell_counts, num_cells_per_dim
):
    """
    Computes forces on each particle using the linked-cell algorithm, optimized with numba. Needs to be conducted outside of the Simulation class.
    Since numba works fastest on loops, nested-loops instead of matrix multiplications are applied. 


    Parameters
    ----------
    positions : arr
        Array of particle positions in natural units.
    boxsize : float
        Size of the simulation box in natural units.
    rcutoff : float
        Cutoff radius for interactions in natural units.
    cell_particles : arr
        Array of particle indices in each cell.
    cell_counts : arr
        Array of particle counts in each cell.
    num_cells_per_dim : int
        Number of cells per dimension.

    Returns
    -------
    forces : arr
        Array of forces on each particle in natural units.
    """

    N = positions.shape[0]
    forces = np.zeros_like(positions)

    for cx in range(num_cells_per_dim):
        for cy in range(num_cells_per_dim):
            for cz in range(num_cells_per_dim):

                cell_id = cx + num_cells_per_dim * (cy + num_cells_per_dim * cz)

                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):

                            nx = (cx + dx) % num_cells_per_dim
                            ny = (cy + dy) % num_cells_per_dim
                            nz = (cz + dz) % num_cells_per_dim

                            neighbor_id = nx + num_cells_per_dim * (
                                ny + num_cells_per_dim * nz
                            )
                            
                            if neighbor_id < cell_id: ## Avoid double counting of cells
                                continue

                            for a in range(cell_counts[cell_id]):
                                i = cell_particles[cell_id, a]
                                if i == -1:
                                    continue

                                for b in range(cell_counts[neighbor_id]):
                                    j = cell_particles[neighbor_id, b]

                                    if neighbor_id == cell_id:
                                        if j <= i:
                                            continue

                                    rij = positions[j] - positions[i]
                                    
                                    for k in range(3):
                                        if rij[k] > 0.5 * boxsize:
                                            rij[k] -= boxsize
                                        elif rij[k] < -0.5 * boxsize:
                                            rij[k] += boxsize

                                    dist2 = rij[0] ** 2 + rij[1] ** 2 + rij[2] ** 2

                                    if dist2 < rcutoff * rcutoff:
                                        dist = np.sqrt(dist2)
                                        fmag = -interaction_force(dist)

                                        for k in range(3):
                                            f = fmag * rij[k]
                                            forces[i, k] += f
                                            forces[j, k] -= f

    return forces
