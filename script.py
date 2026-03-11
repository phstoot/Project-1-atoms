from restructure_setup import Simulation
import matplotlib.pyplot as plt
import numpy as np

test = Simulation()

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test.positions[:,0], test.positions[:,1], test.positions[:,2])
ax.set_aspect('equal')
ax.set_title('initial positions')
plt.show()

# fig = plt.figure(figsize=(6,6))
# plt.hist(test.velocities.flatten(), bins=20)
# plt.title('initial velocity distribution')
# plt.show()


test.run()
x = np.arange(1000)
fig2 = plt.figure(figsize=(12, 4))
plt.plot(x, test.e_kin_hist, label='kin')
plt.plot(x, test.e_pot_hist, label ='pot')
plt.title('energy evolution')
plt.legend()
plt.show()

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test.positions[:,0], test.positions[:,1], test.positions[:,2])
ax.set_aspect('equal')
ax.set_title('end positions')
plt.show()

