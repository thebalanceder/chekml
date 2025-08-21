import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL("./DISOwithRCF.so")

# Define the C function pointer type for the objective function
OBJ_FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double))

# Define structures for C integration
class Solution(ctypes.Structure):
    _fields_ = [("position", ctypes.c_double * 12), 
                ("fitness", ctypes.c_double)]

class DISO_Optimizer(ctypes.Structure):
    _fields_ = [("dim", ctypes.c_int),
                ("population_size", ctypes.c_int),
                ("max_iter", ctypes.c_int),
                ("bounds", ctypes.c_double * 20),
                ("best_solution", Solution)]

# Function prototypes
lib.diso_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
lib.diso_init.restype = ctypes.POINTER(DISO_Optimizer)

lib.diso_optimize.argtypes = [ctypes.POINTER(DISO_Optimizer), OBJ_FUNC_TYPE]
lib.diso_free.argtypes = [ctypes.POINTER(DISO_Optimizer)]

# Define Python wrapper class
class DISOWrapper:
    def __init__(self, dim, population_size, max_iter, bounds):
        bounds_arr = (ctypes.c_double * (dim * 2))(*np.array(bounds).flatten())
        self.obj = lib.diso_init(dim, population_size, max_iter, bounds_arr)

    def optimize(self, objective_function):
        # Convert Python function to C function pointer
        obj_func = OBJ_FUNC_TYPE(objective_function)
        lib.diso_optimize(self.obj, obj_func)

        # Collect history from optimizer
        history = []
        for i in range(self.obj.contents.max_iter):
            best_pos = self.obj.contents.best_solution.position[:self.obj.contents.dim]
            history.append(tuple(best_pos))  # Convert to tuple for Python safety
        
        return history

    def get_best_solution(self):
        return self.obj.contents.best_solution.position[:self.obj.contents.dim], self.obj.contents.best_solution.fitness

    def free(self):
        lib.diso_free(self.obj)
