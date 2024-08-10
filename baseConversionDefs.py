import numpy as np

def bC10toN(cf, bf, ni_copy, expo, factor):
    """Calculate exponents and factors needed for converting a number in base 10 to another base N."""
    for i in cf:
        if (np.log(bf) != 0) & (int(ni_copy)>=1):   #se log definito e numero non nullo
            exp = int(np.log(int(ni_copy))/np.log(bf))      #calcola esponenete nella base finale
            if (exp>=0) & ((int(ni_copy) - int(cf[-1-i]*(bf**exp)))>=0):    #se esponente e numero sottratto non negativi 
                expo.append(exp)                                        
                factor.append(cf[-1-i])
                ni_copy -= cf[-1-i]*(bf**exp)   #aggiunge esponente e fattore alle rispettive liste e aggiorna numero
                break
            else:
                i+=1
    return(ni_copy)


def bCNto10(bi, ni):
    """Converts a number from a generic base N to base 10."""
    ni10=0
    for i in range(0, len(ni)):
        if ord(ni[i]) >= 55:  #acts on digits (from base>37)
            ni10+= (int(ord(ni[i])-55))*bi**(len(ni)-i-1)        
        elif ord(ni[i]) >= 55:  #acts on digits (from base>10)
            ni10+= (int(ord(ni[i])-55))*bi**(len(ni)-i-1)   
        else:     #acts on digits (base 10 max)
            ni10+= int(ni[i])*bi**(len(ni)-i-1)     
    return (ni10)
