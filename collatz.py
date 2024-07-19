import matplotlib.pyplot as plt

n = int(input("Insert the initial number: "))
m = int(input("\nDo you want to try the modified version? [Type 0 for NO and 1 for YES] "))

x = []
y = []
i = 0

y.append(n)
x.append(0)

if m == 0:
    while (n!=1):
        if (n%2)==0:
            n=n/2
        else:
            n=3*n+1 
        i+=1
        x.append(i)
        y.append(n)
        print(n)

else:
    while n!=4:
        if (n%3)==0:
            n=n/3
        elif (n%3)==1:
            n=5*n+1
        else:
            n=5*n+2
        i+=1
        x.append(i)
        y.append(n)
        print(n)
    
    

plt.plot(x, y)
plt.xlabel("number of iteration")
plt.ylabel("number reached")

if m == 0:
    plt.title("Collatz's Conjecture")
else:
    plt.title("Modified Collatz's Conjecture")
plt.show()





