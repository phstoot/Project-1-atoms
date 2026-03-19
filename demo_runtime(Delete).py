from cpa_group5.simulation import Simulation
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
from cpa_group5.utils import spacer, section
import matplotlib as mpl
from cycler import cycler
import copy
import time
import multiprocessing as mp




# Measuring the runtime for each combination of algorithm and number of particles, and comparing the results.

def run_with_timeout(func, timeout=300):
    def wrapper(queue):
        start = time.time()
        func()
        queue.put(time.time() - start)

    queue = mp.Queue()
    p = mp.Process(target=wrapper, args=(queue,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return np.nan  # timeout

    return queue.get()
  
def main():
    alg = 'verlet'

    FCC_number = [(4*n**3) for n in range(3, 10)]
    normal_runtimes = []
    linked_cell_runtimes = []
    linked_cell_numba_runtimes = []

    for num_particles in FCC_number:
        print(f"Testing with {num_particles} particles.")
        
        default_sim = Simulation(
            num_particles=num_particles,
            density=1.2,
            temp=0.5,
            optimized=True,
            numba=True
        )

        default_sim.equilibrate()

        normal_sim = copy.deepcopy(default_sim)
        normal_sim.optimized = False

        linked_cell_sim = copy.deepcopy(default_sim)
        linked_cell_sim.numba = False

        linked_cell_numba_sim = copy.deepcopy(default_sim)

        print("Normal algorithm")
        normal_runtime = run_with_timeout(lambda: normal_sim.run(steps=500))
        normal_runtimes.append(normal_runtime)
        print("Runtime:", normal_runtime)

        print("Linked-cell (NumPy)")
        linked_cell_runtime = run_with_timeout(lambda: linked_cell_sim.run(steps=500))
        linked_cell_runtimes.append(linked_cell_runtime)
        print("Runtime:", linked_cell_runtime)

        print("Linked-cell (Numba)")
        linked_cell_numba_runtime = run_with_timeout(lambda: linked_cell_numba_sim.run(steps=500))
        linked_cell_numba_runtimes.append(linked_cell_numba_runtime)
        print("Runtime:", linked_cell_numba_runtime)

        spacer()



    print(
        f"{'num_particles':<10} {r'Runtime Normal':<20} {r'Runtime LC':<20} {r'Runtime LC+Numba':<20} \n"
    )
    print("-" * 65 + "\n")
    for step, nump_particles in enumerate(FCC_number):
        print(
            f"{num_particles:<10} {normal_runtimes[step]:<20.4f} {linked_cell_runtimes[step]:<20.4f} {linked_cell_numba_runtimes[step]:<20.4f} \n"
        )
    print("Done.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()