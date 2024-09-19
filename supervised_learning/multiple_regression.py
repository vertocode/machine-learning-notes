from utils import get_values_by_index
import numpy as np
import time

# liner regression with multiple features
# Fw,b(x) = (w1*x1) + (w2*x2) + (w3*x3) + ... +  (wn*xn) + b
# The same logic but with different syntax:
# Fw(vector),b(x(vector)) = w(vector) . x(vector) + b

# feature
houses = [1, 1]
floors = [1, 2]
bedrooms = [1, 3]
years = [1, 4]
x_train = np.array([houses, floors, bedrooms, years])

# xj
xj = x_train[1]
# x(i)
xi = get_values_by_index(x_train, 1)
# xj(i)
xji = x_train[1][1]

def process_without_vectorization(x, w, b):
    n = len(x)
    result = 0
    for index in range(n):
        result = result + w[index] * x[index]
    result = result + b
    return result

def process_with_vectorization(x, w, b):
    return np.dot(w, x) + b

x_vector = np.arange(10000000)
w_vector = np.arange(10000000)

# Measure time for process_without_vectorization
start_time_without_vectorization = time.time()
result_without_vectorization = process_without_vectorization(x_vector, w_vector, 5)
end_time_without_vectorization = time.time()
time_without_vectorization = end_time_without_vectorization - start_time_without_vectorization

# Measure time for process_with_vectorization
start_time_with_vectorization = time.time()
result_with_vectorization = process_with_vectorization(x_vector, w_vector, 5)
end_time_with_vectorization = time.time()
time_with_vectorization = end_time_with_vectorization - start_time_with_vectorization

print(result_without_vectorization, 'without vectorization')
print(result_with_vectorization, 'with vectorization')
print(f"Time without vectorization: {time_without_vectorization} seconds")
print(f"Time with vectorization: {time_with_vectorization} seconds")


