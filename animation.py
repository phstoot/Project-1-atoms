from simulate_verlet import verlet_integration
from simulate_euler import simulate

## !!! doesn't work yet properly, work on generalizing initial variables and 
# position & velocity arrays.
colors = []
for i in range(N): # from some stackexchange post
    colors.append('#%06X' % randint(0, 0xFFFFFF))


# Animation part
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
ax.set_title("Verlet Algorithm")

# Create scatter object (this is what we update)
scat = ax.scatter(pos[:,0], pos[:,1], c=colors, s=80)

def update(frame):
    verlet_integration()  # advance system by one step
    scat.set_offsets(pos)  # update particle positions
    return scat,

ani = animation.FuncAnimation(
    fig,
    update,
    frames=1000,
    interval=20,
    blit=True,
    repeat=True
)

plt.show()