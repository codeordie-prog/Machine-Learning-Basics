"""
Familiarize with basic numpy functionality

"""

import numpy as np

# Memory usage
def compare_memory(size):
    
    d_types = [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]
    for dt in d_types:
        
        if str(np.dtype(dt)).startswith("float"):
            arr = np.random.rand(size).astype(dt)
        else:
            arr = np.random.randint(0, 100, size, dtype=dt)
        
        print(f"{dt.__name__}: size: {arr.nbytes / (1024 * 1024):.2f} MB")


# Indexing

arr = np.arange(10)
x = arr[0]
print(x)
print(type(x)) # <class 'numpy.int32'> x is a numpy scalar, not a python int



# Element wise operations
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])

c = np.random.randint(-5, 6, size=2)
print(c)
print(f"Binary operators work element wise: {a + b}")

# Scalar operations
print(f"Scalar operations: {a * 2}")
print(f"Scalar operations: {a[1] * 2}")


# Matrices

d = np.array([1,2,3,4,8,9])
e = np.array([1,2,4,5,6,7])

res = np.multiply(d, e, out=np.empty_like(d))
print(res)