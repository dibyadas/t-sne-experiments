import numpy as np
# import time
import timeit

num_rows = 1000
num_feature = 2
y = np.random.random((num_rows, num_feature))

def my_function_direct(y):
    b = np.tile(y, (num_rows,1,1))
    bT = b.transpose((1,0,2))
    Ddir = np.sum(np.square(b-bT), axis=2)
    return Ddir

def my_function_chatgpt(y):
    diff_matrix = np.sum((y[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2, axis=2)
    return diff_matrix

def my_function_paper(y):
    sumy = np.sum(np.square(y), 1)
    num = -2. * np.dot(y, y.T)
    num = np.add(np.add(num, sumy).T, sumy)
    return num    

# Timing the function with timeit
elapsed_time1 = timeit.timeit(lambda : my_function_direct(y), number=1000)
print(f"Elapsed time: {elapsed_time1} seconds")

elapsed_time2 = timeit.timeit(lambda : my_function_chatgpt(y), number=1000)
print(f"Elapsed time: {elapsed_time2} seconds")

elapsed_time3 = timeit.timeit(lambda : my_function_paper(y), number=1000)
print(f"Elapsed time: {elapsed_time3} seconds")




# np.dot(y, y.T)