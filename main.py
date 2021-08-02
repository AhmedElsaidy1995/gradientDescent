import numpy as np

def f(x):
    return x**3 - 7*x**2 + 8*x - 3

def df(x):
    return 3*x**2 -14*x + 8

def newton(f,df,p,tolerance):
    count = 1
    x1 = p
    x2 = x1 - f(x1) / df(x1)
    while abs(x2 - x1) > tolerance:
        x1 = x2
        x2 = x1 - f(x1) / df(x1)
        count += 1
    print(x2)
    return x2, count

result,iterations = newton(f,df,1,1e-5)
print("result:",result)
print("Number of Iterations:",iterations)
