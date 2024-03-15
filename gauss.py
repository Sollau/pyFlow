import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

x_min = 0.0
x_max = 16.0

mean = 8.0
std = 2.0

x= np.linspace(x_min,x_max, 100)
y = scipy.stats.norm.pdf(x, mean, std)

plt.plot(x, y, color='coral')
plt.xlim(x_min, x_max)
plt.ylim(0, 0.25)
plt.xlabel('x')
plt.ylabel('Normal Distribution')

plt.savefig("gauss.png")
plt.show()
