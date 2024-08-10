import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")

dims = 1
step_n = 10000
step_set = [-1,0,1]
origin = np.zeros((1,dims))

step_shape = (step_n, dims)
steps = np.random.choice(a=step_set, size=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
start =path[:1]
stop = path[-1:]

fig = plt.figure(figsize=(6,4), dpi=200)
ax = fig.add_subplot(111)
ax.set_xlabel('N steps')
ax.set_ylabel('X')

ax.scatter(np.arange(step_n+1), path, c='blue', alpha = 0.25, s=0.05) #plot points+connecting lines

ax.plot(0, start, c='red', marker='+')
ax.plot(step_n, stop, c='black', marker = 'o')

plt.tight_layout(pad=0)
plt.savefig('randomwalk_1D.png', dpi=250)
plt.show()