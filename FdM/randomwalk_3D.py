import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")

dims = 3
step_n = 1000
step_set = [-1,0,1]
origin = np.zeros((1,dims))

step_shape = (step_n, dims)
steps = np.random.choice(a=step_set, size=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
start =path[:1]
stop = path[-1:]

fig = plt.figure(figsize=(10,10), dpi=200)
ax = fig.add_subplot(111, projection='3d')
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.scatter3D(path[:,0], path[:,1], path[:,2], c='blue', alpha = 0.25, s=1) #plot points
ax.plot3D(path[:,0], path[:,1], path[:,2], c='blue', alpha = 0.5, lw=0.25) #plot connecting lines

ax.plot3D(start[0, 0], start[0, 1], start[0, 2], c='red', marker='+')
ax.plot3D(stop[0,0], stop[0,1], path[0,2], c='black', marker = 'o')

plt.tight_layout(pad=0)
plt.savefig('randomwalk_3D.png', dpi=250)
plt.show()