import baseConversionFunctions as bC

ni = input("Insert your number: ")
bi = int(input("Indicate the current base (must be < 63): "))
bf = int(input("Indicate the base you want to convert your number to (must be < 63): "))
ndigits=int(input("\nInsert the extimated number of digits you'll need in the new base: "))

assert (bi<63), "Initial base is too big" #abortisce l'esecuzione se la base è troppo grande
assert (bf<63), "Final base is too big"

xi = len(str(ni)) # quantità di cifre del numero iniziale

ni10 = bC.bCNto10(bi, ni)

ci = [] #cifre della base iniziale
for i in range(bi):
    if i>35:
        ci.append(chr(61+i))
    elif i>9:
        ci.append(chr(55+i))
    else:
        ci.append(i)

ci10 = list(range(0,bi)) #cifre della base iniziale espresse in base 10
        
cf = [] #cifre della base finale
for i in range(bf): 
    if i>35:
        cf.append(chr(61+i))
    elif i>9:
        cf.append(chr(55+i))
    else:
        cf.append(i)
        
cf10= list(range(0,bf)) #cifre della base finale espresse in base 10

addFactor10 = False

expo=[]     #elenco esponenti necessari
factor10=[]     #elenco fattori necessari (cifre in base 10)
ni_copy=[ni10]

nf10=0      #numero finale (sse tutte cifre in base 10)
for i in range(0,ndigits,1):
    ni_copy.append(bC.bC10toN(cf10, bf, ni_copy[i], expo, factor10))    
    if (i < len(factor10)) & (i<len(expo)):
        nf10 += factor10[i]*10**expo[i]

extExpor=[]     #elenco degli esponenti necessari con aggiunta di segnaposti vuoti (ordine crescente)
for i in range(0, expo[0]+1):
    if i in expo:
        extExpor.append(i)
    else:
        extExpor.append("")

extExpo = extExpor[::-1]    #elenco degli esponenti necessari con aggiunta di segnaposti vuoti (ordine decrescente)
        
extFactor10=[] #elenco dei fattori necessari con aggiunta di segnaposti vuoti (con esponente decrescente)
j=0
for i in range (0, len(extExpo)): 
    if extExpo[i] in expo:
        extFactor10.append(factor10[j])
        j+=1
    else:
        extFactor10.append(0)

factor=[]   #elenco fattori necessari espressi in cifre della base voluta
for i in range(0,len(extExpo)):
    if extFactor10[i]>35:
        factor.append(chr(61+extFactor10[i]))
        addFactor10=True
    elif extFactor10[i]>9:
        factor.append(chr(55+extFactor10[i]))
        addFactor10=True
    else:
        factor.append(extFactor10[i])
    
nfStr =""   #stringa per numero convertito
for i in range(0, len(factor)):
    nfStr+=str(factor[i])   


#control prints and result
print("\nThis is the initial number lenght: " + str(xi))
print("\nThis is your number converted to base 10: " + str(ni10))
print("\nThis are the digits available in your current base: " + str(ci))
if bi>10:
    print(" --> with digits in base-10 only, they're equal to: " + str(ci10)) 

print("\nThis are the digits available in your wanted base: " + str(cf))
if bf>10:
    print(" --> with digits in base-10 only, they're equal to: "+ str(cf10))

print("\nThis is the list of exponents needed: " + str(expo) + " (the extended list is " + str(extExpo)+")")
print("This is the list of factors needed: " + str(factor) + " (the extended list is " + str(extFactor10)+")")

print("CONTROL: "+ str(factor10))
if addFactor10:
    print(" --> with digits in base-10 only, they're equal to: "+ str(factor10))

print("\nCONTROL: This is ni after subsequent subtractions: "+ str(ni_copy))

print("\nThis is your converted number: "+ str(nfStr))
if addFactor10:
    print(" --> with digits in base-10 only, it's equal to: "+ str(nf10))

if len(nfStr) == ndigits:
    print("Congrats! You also extimated well the number of digits you needed\n")
elif len(nfStr)>ndigits:
    print("BAD! You needed more digits... please try again in order to be sure to get the correct number!\n")
else:
    print("You overextimated a bit the number of necessary digits, however, everything's good!\n")