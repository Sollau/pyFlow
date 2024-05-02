import numpy as np

def bCMaxDiv(cf, bf, ni_copy, expo, factor):
    
    for i in cf:
        if (np.log(bf) != 0) & (int(ni_copy)>=1):
            exp = int(np.log(int(ni_copy))/np.log(bf))
            if (exp>=0) & ((int(ni_copy) - cf[-1-i]*(bf**exp))>=0):
                expo.append(exp)
                factor.append(cf[-1-i])
                ni_copy -= cf[-1-i]*(bf**exp)            
                print("Number after subtraction: " + str(ni_copy))
                break
    return(ni_copy)