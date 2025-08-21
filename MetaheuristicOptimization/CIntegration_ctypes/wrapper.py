import ctypes
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for verbose output

# Load shared library
lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "generaloptimizer.so"))

# Define function pointer type
OBJ_FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double))

# Define C structures
class Solution(ctypes.Structure):
    _fields_ = [("position", ctypes.POINTER(ctypes.c_double)), 
                ("fitness", ctypes.c_double)]

class Optimizer(ctypes.Structure):
    _fields_ = [("dim", ctypes.c_int),
                ("population_size", ctypes.c_int),
                ("max_iter", ctypes.c_int),
                ("bounds", ctypes.POINTER(ctypes.c_double)),
                ("population", ctypes.POINTER(Solution)),
                ("best_solution", Solution),
                ("optimize", ctypes.CFUNCTYPE(None, ctypes.c_void_p, OBJ_FUNC_TYPE))]

# Function prototypes
lib.general_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_char_p]
lib.general_init.restype = ctypes.POINTER(Optimizer)

lib.general_optimize.argtypes = [ctypes.POINTER(Optimizer), OBJ_FUNC_TYPE]
lib.general_optimize.restype = None

lib.general_free.argtypes = [ctypes.POINTER(Optimizer)]
lib.general_free.restype = None

lib.get_best_solution.argtypes = [ctypes.POINTER(Optimizer), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
lib.get_best_solution.restype = None

class Wrapper:
    def __init__(self, dim, population_size, max_iter, bounds, method="DISO"):
        """ Initialize the optimizer. """
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.bounds = np.array(bounds, dtype=np.float64).flatten()
        self.method_str = method.encode('utf-8')

        # Convert bounds to C-compatible array
        bounds_arr = (ctypes.c_double * (dim * 2))(*self.bounds)
        
        logging.info(f"🚀 Initializing Optimizer (Method: {method})")
        
        self.obj = lib.general_init(dim, population_size, max_iter, bounds_arr, self.method_str)
        if not self.obj:
            raise RuntimeError("❌ Optimizer initialization failed!")

    def optimize(self, objective_function):
        """ Run the optimization process. """
        if not self.obj:
            raise RuntimeError("❌ Optimizer object is not initialized!")

        obj_func = OBJ_FUNC_TYPE(objective_function)
        logging.info("🚀 Starting Optimization")
        lib.general_optimize(self.obj, obj_func)
        logging.info("✅ Optimization completed!")

    def get_best_solution(self):
        """ Retrieve the best solution directly from C. """
        if not self.obj:
            raise RuntimeError("❌ Optimizer object is not initialized!")

        best_fitness = ctypes.c_double()
        best_position = (ctypes.c_double * self.dim)()

        lib.get_best_solution(self.obj, best_position, ctypes.byref(best_fitness))

        return np.ctypeslib.as_array(best_position, shape=(self.dim,)), best_fitness.value

    def free(self):
        """ Free allocated memory. """
        if self.obj:
            logging.info("🛑 Freeing Optimizer Memory")
            lib.general_free(self.obj)
            self.obj = None
            logging.info("✅ Optimizer memory freed successfully!")
