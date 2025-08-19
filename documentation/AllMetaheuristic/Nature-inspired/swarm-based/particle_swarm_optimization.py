import numpy as np
import matplotlib.pyplot as plt

class ParticleSwarmOptimizer:
    """
    Implementation of Particle Swarm Optimization (PSO) algorithm.

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
    def __init__(self, cost_function, n_var, var_min, var_max, max_it=1000, n_pop=100, 
                 w=1, wdamp=0.99, c1=1.5, c2=2.0):
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
        self.global_best = {'Cost': float('inf'), 'Position': None}
        self.best_costs = np.zeros(max_it)
        self.history = []  # To store global best positions over iterations

    def initialize_swarm(self):
        """Initialize the swarm with random positions and velocities."""
        empty_particle = {
            'Position': None,
            'Cost': None,
            'Velocity': None,
            'Best': {'Position': None, 'Cost': None}
        }
        self.particles = [empty_particle.copy() for _ in range(self.n_pop)]

        for i in range(self.n_pop):
            # Initialize Position
            self.particles[i]['Position'] = np.random.uniform(self.var_min, self.var_max, self.var_size)
            
            # Initialize Velocity
            self.particles[i]['Velocity'] = np.zeros(self.var_size)
            
            # Evaluation
            position_flat = self.particles[i]['Position'].flatten()
            self.particles[i]['Cost'] = float(self.cost_function(position_flat))
            
            # Update Personal Best
            self.particles[i]['Best']['Position'] = self.particles[i]['Position'].copy()
            self.particles[i]['Best']['Cost'] = self.particles[i]['Cost']
            
            # Update Global Best
            if self.particles[i]['Best']['Cost'] < self.global_best['Cost']:
                self.global_best['Position'] = self.particles[i]['Best']['Position'].copy()
                self.global_best['Cost'] = self.particles[i]['Best']['Cost']

    def optimize(self):
        """Run the Particle Swarm Optimization algorithm."""
        self.initialize_swarm()
        
        for it in range(self.max_it):
            for i in range(self.n_pop):
                # Update Velocity
                self.particles[i]['Velocity'] = (self.w * self.particles[i]['Velocity'] +
                    self.c1 * np.random.rand(*self.var_size) * 
                    (self.particles[i]['Best']['Position'] - self.particles[i]['Position']) +
                    self.c2 * np.random.rand(*self.var_size) * 
                    (self.global_best['Position'] - self.particles[i]['Position']))
                
                # Apply Velocity Limits
                self.particles[i]['Velocity'] = np.maximum(self.particles[i]['Velocity'], self.vel_min)
                self.particles[i]['Velocity'] = np.minimum(self.particles[i]['Velocity'], self.vel_max)
                
                # Update Position
                self.particles[i]['Position'] = self.particles[i]['Position'] + self.particles[i]['Velocity']
                
                # Velocity Mirror Effect
                is_outside = (self.particles[i]['Position'] < self.var_min) | (self.particles[i]['Position'] > self.var_max)
                self.particles[i]['Velocity'][is_outside] = -self.particles[i]['Velocity'][is_outside]
                
                # Apply Position Limits
                self.particles[i]['Position'] = np.maximum(self.particles[i]['Position'], self.var_min)
                self.particles[i]['Position'] = np.minimum(self.particles[i]['Position'], self.var_max)
                
                # Evaluation
                position_flat = self.particles[i]['Position'].flatten()
                self.particles[i]['Cost'] = float(self.cost_function(position_flat))
                
                # Update Personal Best
                if self.particles[i]['Cost'] < self.particles[i]['Best']['Cost']:
                    self.particles[i]['Best']['Position'] = self.particles[i]['Position'].copy()
                    self.particles[i]['Best']['Cost'] = self.particles[i]['Cost']
                    
                    # Update Global Best
                    if self.particles[i]['Best']['Cost'] < self.global_best['Cost']:
                        self.global_best['Position'] = self.particles[i]['Best']['Position'].copy()
                        self.global_best['Cost'] = self.particles[i]['Best']['Cost']
            
            # Store global best position in history
            self.history.append(self.global_best['Position'].flatten().copy())
            
            self.best_costs[it] = self.global_best['Cost']
            print(f"Iteration {it + 1}: Best Cost = {self.best_costs[it]}")
            
            # Update Inertia Weight
            self.w *= self.wdamp
        
        return self.global_best, self.best_costs, self.history

# Example usage
def sphere(x):
    """Sphere function for testing the optimizer."""
    return np.sum(x**2)

if __name__ == "__main__":
    # Problem Definition
    cost_function = sphere
    n_var = 10
    var_min = -10
    var_max = 10
    
    # Initialize and Run PSO
    pso = ParticleSwarmOptimizer(cost_function, n_var, var_min, var_max)
    best_solution, best_costs, history = pso.optimize()
    
