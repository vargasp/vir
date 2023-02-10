import numpy as np
import time
from multiprocessing import Pool, RawArray, Array


# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(X, X_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape


def worker_func(i):
    # Simply computes the sum of the i-th row of the input matrix X
    X_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    time.sleep(.1) # Some heavy computations
    return np.asscalar(np.sum(X_np[i,:]))

#Read-only Array 
def mp_unlock(data):
    X_shape = data.shape
    
    # Create a shared array of double precision without a lock.
    X = RawArray('d', data.size)

    # Wrap X as an numpy array so we can easily manipulates its data.
    X_np = np.frombuffer(X).reshape(X_shape)
    
    # Copy data to shared array.
    np.copyto(X_np, data)
    
    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of each worker.
    # (Because X_shape is not a shared variable, it will be copied to each
    # child process.)
    with Pool(processes=4, initializer=init_worker, initargs=(X, X_shape)) as pool:
        result = pool.map(worker_func, range(X_shape[0]))
        print('Results (pool):\n', np.array(result))
        
    # Should print the same results.
    print('Results (numpy):\n', np.sum(X_np, 1))
    
#Writeable Array    
def mp_lock(data):
    X_shape = data.shape
    
    # Create a shared array of double precision with a lock.
    X = Array('d', data.size)    
    
    # Wrap X as an numpy array so we can easily manipulates its data.
    X_np = np.frombuffer(X.get_obj()).reshape(X_shape)
   
    # Copy data to shared array.
    np.copyto(X_np, data)
    
    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of each worker.
    # (Because X_shape is not a shared variable, it will be copied to each
    # child process.)
    with Pool(processes=4, initializer=init_worker, initargs=(X.get_obj(), X_shape)) as pool:
        result = pool.map(worker_func, range(X_shape[0]))
        print('Results (pool):\n', np.array(result))
        
    # Should print the same results.
    print('Results (numpy):\n', np.sum(X_np, 1))

 

# A global dictionary storing the variables passed from the initializer.
var_dicts = {}


def workers_func(i):
    # Simply computes the sum of the i-th row of the input matrix X
    X_np = np.frombuffer(var_dicts['X']).reshape(var_dicts['X'].shape)
    time.sleep(.1) # Some heavy computations
    return np.asscalar(np.sum(X_np[i,:]))


def init_workers(X, Y):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dicts['X'] = X
    var_dicts['Y'] = Y


#Read-only Array and unsahred additional args
def mp_unlock_args(data_arrays):
    
    # Create an aaray of shared arrays of double precision without a lock.
    for i, data in enumerate(data_arrays):
        X = RawArray('d', data.size)
 
    # Wrap X as an numpy array so we can easily manipulates its data.
    X_np = np.frombuffer(X).reshape(data.shape)
    
    # Copy data to shared array.
    np.copyto(X_np, data)
    
    Y = 1
    # Start the process pool and do the computation.
    with Pool(processes=4, initializer=init_workers, initargs=(X, Y)) as pool:
        result = pool.map(workers_func, range(data.shape[0]))
        print('Results (pool):\n', np.array(result))
        
    # Should print the same results.
    print('Results (numpy):\n', np.sum(X_np, 1))
     
 
 