import pylab as pl

x= pl.linspace(0,5,1000)
E = pl.exp(-1/x)/(1+pl.exp(-1/x))

pl.subplot(2,1,1)
pl.plot(x, E)
pl.xlabel(''), pl.ylabel('U / epsilon')

CV = pl.diff(E)/pl.diff(x)
xmid = 0.5*(x[1:]+x[0:-1])

pl.subplot(2,1,2)
pl.plot(xmid, CV)
pl.xlabel('K_B T / epsilon'), pl.ylabel('C_V / K_B')

pl.savefig("2levels.png")
pl.show()