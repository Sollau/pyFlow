from scipy.stats import binom
import matplotlib.pyplot as plt

n = 50
p = 0.4

r_values = list(range(n+1))

mean, var = binom.stats(n,p)

dist = [binom.pmf(r,n,p) for r in r_values]

f = open("binomial.dat", "w")
print("r\tp(r)")
for i in range(n+1):
    print(str(r_values[i])+"\t"+str(dist[i]))
print("mean = "+str(mean))
print("variance = " +str(var))

plt.bar(r_values, dist)
plt.savefig("binomial.png")
plt.show()