"""
Speed test between vectorized code and Non vectorized code
"""

import time
import numpy as np

x = [1, 2, 3, 4, 5] * 10000000
y = [6, 7, 8, 9, 10] * 10000000
np_x = np.array(x)
np_y = np.array(y)

def time_it(fn):
    def enhanced_fn(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print(f"Time taken by {fn.__name__}: {end - start}")
        return result   
    return enhanced_fn

@time_it
def non_vectorized_code():  
    result = 0
    for i in range(len(x)):
        result += x[i] * y[i]
    return result


@time_it
def vectorized_code():
    return np.dot(np_x, np_y)

def main():
    non_vectorized_result = non_vectorized_code()
    vectorized_result = vectorized_code()

    assert non_vectorized_result == vectorized_result, "Results do not match"
    
    print(f"Non vectorized result: {non_vectorized_result}")
    print(f"Vectorized result: {vectorized_result}")

if __name__ == "__main__":
    main()

