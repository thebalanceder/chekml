import ctypes
import numpy as np

# Load the compiled GPC shared library
gpc_lib = ctypes.CDLL("./GPC.so")

# Define the C function pointer type for the objective function
OBJ_FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double))

# Define structures for C integration
class Solution(ctypes.Structure):
    _fields_ = [("position", ctypes.c_double * 10),  # Assuming max dim = 10
                ("fitness", ctypes.c_double)]

class GPC_Optimizer(ctypes.Structure):
    _fields_ = [("dim", ctypes.c_int),
                ("population_size", ctypes.c_int),
                ("max_iter", ctypes.c_int),
                ("bounds", ctypes.c_double * 20),  # Assuming max 10 dimensions (lower+upper bounds)
                ("best_solution", Solution)]

# Function prototypes
gpc_lib.gpc_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
gpc_lib.gpc_init.restype = ctypes.POINTER(GPC_Optimizer)

gpc_lib.gpc_optimize.argtypes = [ctypes.POINTER(GPC_Optimizer), OBJ_FUNC_TYPE]
gpc_lib.gpc_free.argtypes = [ctypes.POINTER(GPC_Optimizer)]

# Define Python wrapper class
class GPCWrapper:
    def __init__(self, dim, population_size, max_iter, bounds):
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter

        # Convert bounds into a contiguous array for C compatibility
        bounds_arr = (ctypes.c_double * (dim * 2))(*np.array(bounds).flatten())
        self.obj = gpc_lib.gpc_init(dim, population_size, max_iter, bounds_arr)

    def optimize(self, func):
        best_solution = None
        best_value = float("inf")
        history = []

        for iteration in range(self.max_iter):
            solutions = self.generate_population()  # Generate candidate solutions
            fitness_values = np.array([func(sol) for sol in solutions])  # Evaluate all candidates

            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < best_value:
                best_value = fitness_values[min_idx]
                best_solution = solutions[min_idx]

            history.append((best_solution.copy(), best_value))  # Store history

            # 🔥 Print progress every 10 iterations 🔥
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best Solution = {best_solution}, Best Fit = {best_value}")

        return history

    def get_best_solution(self):
        """Return best solution and fitness found."""
        return [self.obj.best_solution.position[i] for i in range(self.obj.dim)], self.obj.best_solution.fitness
    def free(self):
        """Free the memory allocated in C."""
        gpc_lib.gpc_free(self.obj)


