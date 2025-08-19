# cython: language_level=3
# distutils: language=c

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY
from libc.stdlib cimport rand, RAND_MAX

# Define NumPy types
DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef np.uint8_t BOOL_t  # For NumPy boolean arrays

cdef class ParticleSwarmOptimizer:
    """
    Cython implementation of Particle Swarm Optimization (PSO) algorithm.

    Parameters:
    - cost_function: Function to optimize (minimize).
    - n_var: Number of decision variables (dimensions).
    - var_min: Lower bound of variables.
    - var_max: Upper bound of variables.
    - max_it: Maximum number of iterations (default: 1000).
    - n_pop: Population size (swarm size, default: 100).
    - w: Inertia weight (default: 1).
    - wdamp: Inertia weight damping ratio (default: 0.99).
    - c1: Personal learning coefficient (default: 1.5).
    - c2: Global learning coefficient (default: 2.0).
    """
    cdef object cost_function
    cdef int n_var, max_it, n_pop
    cdef double var_min, var_max, w, wdamp, c1, c2, vel_max, vel_min
    cdef tuple var_size
    cdef list particles
    cdef dict global_best
    cdef np.ndarray best_costs
    cdef list history

    def __init__(self, cost_function, int n_var, double var_min, double var_max, 
                 int max_it=1000, int n_pop=100, double w=1.0, double wdamp=0.99, 
                 double c1=1.5, double c2=2.0):
        self.cost_function = cost_function
        self.n_var = n_var
        self.var_size = (1, n_var)
        self.var_min = var_min
        self.var_max = var_max
        self.max_it = max_it
        self.n_pop = n_pop
        self.w = w
        self.wdamp = wdamp
        self.c1 = c1
        self.c2 = c2
        self.vel_max = 0.1 * (var_max - var_min)
        self.vel_min = -self.vel_max
        self.particles = None
        self.global_best = {'Cost': INFINITY, 'Position': None}
        self.best_costs = np.zeros(max_it, dtype=DTYPE)
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _initialize_swarm(self):
        """Initialize the swarm with random positions and velocities."""
        cdef int i, j
        cdef double cost
        cdef np.ndarray[DTYPE_t, ndim=2] pos, vel, best_pos
        cdef dict particle

        empty_particle = {
            'Position': None,
            'Cost': None,
            'Velocity': None,
            'Best': {'Position': None, 'Cost': None}
        }
        self.particles = [empty_particle.copy() for _ in range(self.n_pop)]

        for i in range(self.n_pop):
            # Initialize Position
            pos = np.random.uniform(self.var_min, self.var_max, self.var_size).astype(DTYPE)
            self.particles[i]['Position'] = pos
            
            # Initialize Velocity
            vel = np.zeros(self.var_size, dtype=DTYPE)
            self.particles[i]['Velocity'] = vel
            
            # Evaluation
            cost = self.cost_function(pos.flatten())
            self.particles[i]['Cost'] = float(cost)
            
            # Update Personal Best
            self.particles[i]['Best']['Position'] = pos.copy()
            self.particles[i]['Best']['Cost'] = float(cost)
            
            # Update Global Best
            if self.particles[i]['Best']['Cost'] < self.global_best['Cost']:
                self.global_best['Position'] = pos.copy()
                self.global_best['Cost'] = float(cost)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Particle Swarm Optimization algorithm."""
        cdef int it, i, j
        cdef double cost, r1, r2
        cdef np.ndarray[DTYPE_t, ndim=2] pos, vel, best_pos, global_best_pos
        cdef np.ndarray[DTYPE_t, ndim=2] rand1, rand2
        cdef np.ndarray[BOOL_t, ndim=2] is_outside  # Changed to NumPy bool array

        self._initialize_swarm()
        
        for it in range(self.max_it):
            for i in range(self.n_pop):
                pos = self.particles[i]['Position']
                vel = self.particles[i]['Velocity']
                best_pos = self.particles[i]['Best']['Position']
                global_best_pos = self.global_best['Position']
                
                # Update Velocity
                rand1 = np.random.rand(1, self.n_var).astype(DTYPE)
                rand2 = np.random.rand(1, self.n_var).astype(DTYPE)
                vel[...] = (self.w * vel +
                           self.c1 * rand1 * (best_pos - pos) +
                           self.c2 * rand2 * (global_best_pos - pos))
                
                # Apply Velocity Limits
                vel[...] = np.maximum(vel, self.vel_min)
                vel[...] = np.minimum(vel, self.vel_max)
                
                # Update Position
                pos[...] = pos + vel
                
                # Velocity Mirror Effect
                is_outside = (pos < self.var_min) | (pos > self.var_max)
                for j in range(self.n_var):
                    if is_outside[0, j]:
                        vel[0, j] = -vel[0, j]
                
                # Apply Position Limits
                pos[...] = np.maximum(pos, self.var_min)
                pos[...] = np.minimum(pos, self.var_max)
                
                # Evaluation
                cost = self.cost_function(pos.flatten())
                self.particles[i]['Cost'] = float(cost)
                
                # Update Personal Best
                if cost < self.particles[i]['Best']['Cost']:
                    self.particles[i]['Best']['Position'] = pos.copy()
                    self.particles[i]['Best']['Cost'] = float(cost)
                    
                    # Update Global Best
                    if cost < self.global_best['Cost']:
                        self.global_best['Position'] = pos.copy()
                        self.global_best['Cost'] = float(cost)
            
            # Store global best position in history
            self.history.append(self.global_best['Position'].flatten().copy())
            
            self.best_costs[it] = self.global_best['Cost']
            print(f"Iteration {it + 1}: Best Cost = {self.best_costs[it]}")
            
            # Update Inertia Weight
            self.w *= self.wdamp
        
        return self.global_best, self.best_costs, self.history
