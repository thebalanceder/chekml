# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs

# Define NumPy types for Cython
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

class GalacticSwarmOptimizer:
    def __init__(self, objective_function, int dim, bounds, int pop_size=5, int subpop=10, 
                 int epoch_number=5, int iteration1=198, int iteration2=1000, int trials=50):
        """
        Initialize the Galactic Swarm Optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - pop_size: Number of particles in each subswarm.
        - subpop: Number of subswarms (galaxies).
        - epoch_number: Number of epochs for optimization.
        - iteration1: Number of iterations for Level 1 (subswarm optimization).
        - iteration2: Number of iterations for Level 2 (galaxy optimization).
        - trials: Number of independent runs.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)  # Bounds as [(low, high), ...]
        self.pop_size = pop_size
        self.subpop = subpop
        self.epoch_number = epoch_number
        self.iteration1 = iteration1
        self.iteration2 = iteration2
        self.trials = trials
        self.vmin = self.bounds[:, 0]
        self.vmax = self.bounds[:, 1]
        self.xmin = self.bounds[:, 0]
        self.xmax = self.bounds[:, 1]
        self.results = []
        self.history = []  # To store (trial, trial_history)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_swarm(self):
        """ Initialize position and velocity vectors for all subswarms """
        cdef int num, p
        cdef np.ndarray[DTYPE_t, ndim=2] x, v, pbest
        cdef np.ndarray[DTYPE_t, ndim=1] xgbest, cost, pbest_c
        cdef double cgbest
        cdef np.ndarray[DTYPE_t, ndim=1] galaxy_x
        cdef double galaxy_c
        gal = []

        for num in range(self.subpop):
            subswarm = []
            x = np.random.uniform(self.xmin, self.xmax, (self.pop_size, self.dim))
            v = np.random.uniform(self.vmin, self.vmax, (self.pop_size, self.dim))
            pbest = np.random.uniform(self.xmin, self.xmax, (self.pop_size, self.dim))
            cost = np.zeros(self.pop_size, dtype=DTYPE)
            pbest_c = np.zeros(self.pop_size, dtype=DTYPE)

            for p in range(self.pop_size):
                cost[p] = self.objective_function(x[p])
                pbest_c[p] = self.objective_function(pbest[p])
                if cost[p] < pbest_c[p]:
                    pbest[p] = x[p].copy()
                    pbest_c[p] = cost[p]
                
                subswarm.append({
                    'x': x[p],
                    'v': v[p],
                    'pbest': pbest[p],
                    'cost': cost[p],
                    'pbest_c': pbest_c[p]
                })
            
            xgbest = pbest[0].copy()
            cgbest = pbest_c[0]
            for p in range(1, self.pop_size):
                if pbest_c[p] < cgbest:
                    xgbest = pbest[p].copy()
                    cgbest = pbest_c[p]
            
            gal.append({
                'particles': subswarm,
                'xgbest': xgbest,
                'cgbest': cgbest
            })
        
        galaxy_x = gal[0]['xgbest'].copy()
        galaxy_c = self.objective_function(galaxy_x)
        for num in range(1, self.subpop):
            if self.objective_function(gal[num]['xgbest']) < galaxy_c:
                galaxy_x = gal[num]['xgbest'].copy()
                galaxy_c = self.objective_function(gal[num]['xgbest'])
        
        return gal, galaxy_x, galaxy_c

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def level1_optimization(self, gal, np.ndarray[DTYPE_t, ndim=1] galaxy_x, double galaxy_c, int count_f):
        """ Level 1: Optimize within each subswarm """
        cdef int num, r, p
        cdef double c1, c2, r1, r2
        cdef np.ndarray[DTYPE_t, ndim=1] v, x_new

        for num in range(self.subpop):
            for r in range(self.iteration1 + 1):
                c1 = 2.05 * np.random.rand()
                c2 = 2.05 * np.random.rand()
                for p in range(self.pop_size):
                    r1 = -1 + 2 * np.random.rand()
                    r2 = -1 + 2 * np.random.rand()
                    
                    v = c1 * r1 * (gal[num]['particles'][p]['pbest'] - gal[num]['particles'][p]['x']) + \
                        c2 * r2 * (gal[num]['xgbest'] - gal[num]['particles'][p]['x'])
                    
                    gal[num]['particles'][p]['v'] = (1 - r / (self.iteration1 + 1)) * gal[num]['particles'][p]['v'] + v
                    gal[num]['particles'][p]['v'] = np.clip(gal[num]['particles'][p]['v'], self.vmin, self.vmax)
                    x_new = gal[num]['particles'][p]['x'] + gal[num]['particles'][p]['v']
                    gal[num]['particles'][p]['x'] = np.clip(x_new, self.xmin, self.xmax)
                    
                    gal[num]['particles'][p]['cost'] = self.objective_function(gal[num]['particles'][p]['x'])
                    count_f += 1
                    
                    if gal[num]['particles'][p]['cost'] < gal[num]['particles'][p]['pbest_c']:
                        gal[num]['particles'][p]['pbest'] = gal[num]['particles'][p]['x'].copy()
                        gal[num]['particles'][p]['pbest_c'] = gal[num]['particles'][p]['cost']
                        
                        if gal[num]['particles'][p]['pbest_c'] < gal[num]['cgbest']:
                            gal[num]['xgbest'] = gal[num]['particles'][p]['pbest'].copy()
                            gal[num]['cgbest'] = gal[num]['particles'][p]['pbest_c']
                            
                            if gal[num]['cgbest'] < galaxy_c:
                                galaxy_x = gal[num]['xgbest'].copy()
                                galaxy_c = gal[num]['cgbest']
        
        return gal, galaxy_x, galaxy_c, count_f

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def level2_optimization(self, gal, np.ndarray[DTYPE_t, ndim=1] galaxy_x, double galaxy_c, int count_f):
        """ Level 2: Optimize across subswarms """
        cdef int r, p
        cdef double c3, c4, r3, r4
        cdef np.ndarray[DTYPE_t, ndim=1] v, x_new
        particles = []

        for p in range(self.subpop):
            particle = {
                'x': gal[p]['xgbest'].copy(),
                'cost': gal[p]['cgbest'],
                'pbest': gal[p]['xgbest'].copy(),
                'pbest_c': gal[p]['cgbest'],
                'v': gal[p]['xgbest'].copy()
            }
            particles.append(particle)
        
        for r in range(self.iteration2 + 1):
            c3 = 2.05 * np.random.rand()
            c4 = 2.05 * np.random.rand()
            for p in range(self.subpop):
                r3 = -1 + 2 * np.random.rand()
                r4 = -1 + 2 * np.random.rand()
                
                v = c3 * r3 * (particles[p]['pbest'] - particles[p]['x']) + \
                    c4 * r4 * (galaxy_x - particles[p]['x'])
                
                particles[p]['v'] = (1 - r / (self.iteration2 + 1)) * particles[p]['v'] + v
                particles[p]['v'] = np.clip(particles[p]['v'], self.vmin, self.vmax)
                x_new = particles[p]['x'] + particles[p]['v']
                particles[p]['x'] = np.clip(x_new, self.xmin, self.xmax)
                
                particles[p]['cost'] = self.objective_function(particles[p]['x'])
                count_f += 1
                
                if particles[p]['cost'] < particles[p]['pbest_c']:
                    particles[p]['pbest'] = particles[p]['x'].copy()
                    particles[p]['pbest_c'] = particles[p]['cost']
                    
                    if particles[p]['pbest_c'] < galaxy_c:
                        galaxy_x = particles[p]['pbest'].copy()
                        galaxy_c = particles[p]['pbest_c']
        
        return galaxy_x, galaxy_c, count_f

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Galactic Swarm Optimization """
        cdef int trial, epoch, count_f = 0
        cdef np.ndarray[DTYPE_t, ndim=1] galaxy_x
        cdef double galaxy_c = 0.0
        gal = []
        trial_history = []

        for trial in range(self.trials):
            gal, galaxy_x, galaxy_c = self.initialize_swarm()
            count_f = 0
            trial_history = []
            
            for epoch in range(self.epoch_number):
                gal, galaxy_x, galaxy_c, count_f = self.level1_optimization(gal, galaxy_x, galaxy_c, count_f)
                galaxy_x, galaxy_c, count_f = self.level2_optimization(gal, galaxy_x, galaxy_c, count_f)
                trial_history.append((epoch, galaxy_x.copy(), galaxy_c))
                print(f"Epoch={epoch + 1} objfun_val={galaxy_c:.6e}")
            
            print(f"Trial={trial + 1} objfun_val={galaxy_c:.6e}")
            self.results.append(galaxy_c)
            self.history.append((trial, trial_history))
        
        obj_mean = np.mean(self.results)
        obj_std = np.std(self.results)
        obj_var = np.var(self.results)
        obj_median = np.median(self.results)
        # Handle mode calculation for potentially negative values
        if len(self.results) > 0:
            rounded_results = np.round(self.results, 6)
            shift = abs(np.min([0, np.min(rounded_results)])) + 1  # Shift to ensure non-negative
            shifted_results = (rounded_results + shift) * 1e6  # Scale to integers
            obj_mode = (float(np.bincount(shifted_results.astype(int)).argmax()) / 1e6) - shift
        else:
            obj_mode = 0.0
        print(f"\nobj_mean={obj_mean:.6e}")
        print(f"obj_std={obj_std:.6e}")
        print(f"obj_var={obj_var:.6e}")
        print(f"best_val={min(self.results):.6e}")
        print(f"worst_val={max(self.results):.6e}")
        print(f"median={obj_median:.6e}")
        print(f"mode={obj_mode:.6e}")
        print(f"function_calls={count_f}")
        
        return galaxy_x, galaxy_c, self.results, self.history

# Example usage
if __name__ == "__main__":
    def rosenbrock(x):
        """ Rosenbrock function for testing """
        return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    
    dim = 10
    bounds = [(-30, 30)] * dim
    optimizer = GalacticSwarmOptimizer(rosenbrock, dim, bounds, pop_size=5, subpop=10, 
                                       epoch_number=5, iteration1=198, iteration2=1000, trials=50)
    best_solution, best_value, results, history = optimizer.optimize()
