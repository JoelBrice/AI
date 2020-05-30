# Create 2 new lists height and weight
height = [1.87,  1.87, 1.82, 1.91, 1.90, 1.85]
weight = [81.65, 97.52, 95.25, 92.98, 86.18, 88.45]

# Import the numpy package as np
# import numpy as np

# Create 2 numpy arrays from height and weight
# np_height = np.array(height)
# np_weight = np.array(weight)


def print_odd():
    for i in range(10):
        if i%2 !=0:
            print(i)

# finally keyword

def f():
    try:
        1/0
    finally:
        return 42

f()

print_odd()
def func():
    pass

#Coroutine function

async def funct(a,b):
    if a>b:
        print (a)
        await "Hello there"


def arrays(n):
    arr = [3,4,1,5]
    return arr.append(n)

print(arrays(2))



        


