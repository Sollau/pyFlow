import numpy as np
import baseConversionMaxDiv as f

ni = int(input("Insert your number: "))
bi = int(input("Indicate the current base: "))
bf = int(input("Indicate the base you want to convert it to: "))

xi = len(str(ni)) # quantità di cifre del numero iniziale

ci = [] #cifre della base iniziale
for i in range(bi):
    ci.append(i)

cf = [] #cifre della base finale
for i in range(bf): 
     cf.append(i)

expo =[]
factor=[]

print("\n")

ni_copy=[ni]

nf=0

for i in range(0,10,1):                                             #calcola 10 cifre del numero convertito
    ni_copy.append(f.bCMaxDiv(cf, bf, ni_copy[i], expo, factor))    #solo in base 10??
    if (i < len(factor)) & (i<len(expo)):
        nf += factor[i]*bf**expo[i]




print("\nThis is the initial number lenght: " + str(xi))
print("This are the digits available in your current base: " + str(ci))
print("This are the digits available in your wanted base: " + str(cf))

print("This is the list of exponents needed: " + str(expo))
print("This is the list of factors needed: " + str(factor))
print("This is ni after subtraction: "+ str(ni_copy))

print("\nThis is your converted number: " + str(nf))