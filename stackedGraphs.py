import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 10, 10)

for k in range(0,10):
    if x[k]<=5:    
        f = 2*x+k
        plt.plot(x, f, label = "k = "  +str(k))
    else:
        f= -2*x+k
        plt.plot(x, f, label = "k = "+str(k))

plt.legend()
plt.show()