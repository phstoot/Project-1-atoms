import numpy as np
import matplotlib.pyplot as plt
from random import randint

## We could technically write all equations allowing for both natural and normal units (so using function(*,natural = True))

rho = ...  # Particle density, not necessary for now
T = ...  # temperature
epsilon = 1.654 * 10 ** (-10)  # 119.8 #K (epsilon / k_boltzmann)
sigma = 3.405 * 10 ** (-10)  # Angstrom
mass = 6.6 * 10 ** (-26)  # Mass


def Interaction_force(r):
    """
    Calculates force in natural units.

    :param r (float): Distance between two particles in natural units.

    :return (float) : Magnitude of Interaction Force
    """

    F = 24 * (2 * r ** (-14) - r ** (-8))
    return F


## Minimal image convenction
def min_vector(part1, part2, L=10, dim=2):
    """Finds smallest vector connecting particle 1 to particle 2, in the smallest image convention.

    Parameters:
    part1 (arr): main particle
    part2 (arr): interaction particle

    Returns:
    arr: Vector pointing to closest version of interaction particle.
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

    :param vel (arr): Instantaneous velocity array in natural units.

    :return (float): Kinetic Energy array in natural units.
    """

    Kin = 1 / 2 * vel * vel
    return Kin


def Lennard_Jones_Potential(r):
    """
    Calculates Potential Energy due to each particle interaction, in natural units.

    :param r (float): Distance between two particles in natural units.

    :return (float): Interaction Potential
    """

    U = 4 * (r ** (-12) - r ** (-6))
    return U
