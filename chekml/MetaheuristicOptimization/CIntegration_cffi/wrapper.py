import numpy as np
import cffi
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for verbose output

# Initialize CFFI
ffi = cffi.FFI()
lib = ffi.dlopen(os.path.join(os.path.dirname(__file__), "generaloptimizer.so"))

# Define C structures
ffi.cdef("""
    typedef double (*obj_func_type)(double*);

    typedef struct {
        double *position;
        double fitness;
    } Solution;

    typedef struct {
        int dim;
        int population_size;
        int max_iter;
        double *bounds;
        Solution best_solution;
        void (*optimize)(void *, obj_func_type);
    } Optimizer;

    Optimizer* general_init(int dim, int population_size, int max_iter, double *bounds, const char *method);
    void general_optimize(Optimizer *opt, obj_func_type func);
    void general_free(Optimizer *opt);
    void get_best_solution(Optimizer *opt, double *position, double *fitness);
""")

class Wrapper:
    def __init__(self, dim, population_size, max_iter, bounds, method="DISO"):
        """ Initialize the optimizer. """
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.bounds = np.array(bounds, dtype=np.float64).flatten()
        self.method_str = method.encode('utf-8')

        # Convert bounds to C-compatible array
        bounds_arr = ffi.new("double[]", list(self.bounds))
        
        logging.info(f"🚀 Initializing Optimizer (Method: {method})")
        
        self.obj = lib.general_init(dim, population_size, max_iter, bounds_arr, self.method_str)
        if not self.obj:
            raise RuntimeError("❌ Optimizer initialization failed!")

    def optimize(self, objective_function):
        """ Run the optimization process. """
        if not self.obj:
            raise RuntimeError("❌ Optimizer object is not initialized!")

        @ffi.callback("double(double *)")
        def obj_func(c_x):
            x = np.frombuffer(ffi.buffer(c_x, self.dim * ffi.sizeof("double")), dtype=np.float64)
            return objective_function(x)

        logging.info("🚀 Starting Optimization")
        lib.general_optimize(self.obj, obj_func)
        logging.info("✅ Optimization completed!")

    def get_best_solution(self):
        """ Retrieve the best solution directly from C. """
        if not self.obj:
            raise RuntimeError("❌ Optimizer object is not initialized!")

        best_fitness = ffi.new("double *")
        best_position = ffi.new("double[]", self.dim)

        lib.get_best_solution(self.obj, best_position, best_fitness)

        return np.array([best_position[i] for i in range(self.dim)]), best_fitness[0]

    def free(self):
        """ Free allocated memory. """
        if self.obj:
            logging.info("🛑 Freeing Optimizer Memory")
            lib.general_free(self.obj)
            self.obj = None
            logging.info("✅ Optimizer memory freed successfully!")
