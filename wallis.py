def wallis(n):
    """Print the decimals of pi down to n."""
    pi=2
    terms = [4 * i**2 / (4* i**2 - 1) for i in range(n)]
    for element in terms:
        if element == 0:
            continue
        pi = pi * element
    print(pi)
