from restructure_setup import Simulation
import matplotlib.pyplot as plt
import numpy as np

test = Simulation(num_particles=4)

print(test.positions)
diff = test._pairwise_diff_vector_matrix()
dist = np.linalg.norm(diff, axis=-1)
print(dist)

fig = plt.figure(figsize=(6,6))
plt.scatter(test.positions[:,0], test.positions[:,1])
plt.title('initial positions')
plt.show()

test.run()
x = np.arange(1000)
fig2 = plt.figure(figsize=(10, 4))
plt.plot(x, test.e_kin_hist, label='kin')
plt.plot(x, test.e_pot_hist, label ='pot')
plt.title('energy evolution')
plt.legend()
plt.show()

