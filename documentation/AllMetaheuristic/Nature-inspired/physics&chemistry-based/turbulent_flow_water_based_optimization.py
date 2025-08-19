import numpy as np

class TurbulentFlowWaterOptimizer:
    def __init__(self, objective_function, dim, bounds, n_whirlpools=3, n_objects_per_whirlpool=30, max_iter=100):
        """
        Initialize the Turbulent Flow of Water-based Optimization (TFWO) algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - n_whirlpools: Number of whirlpools (groups of solutions).
        - n_objects_per_whirlpool: Number of objects (particles) per whirlpool.
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.n_whirlpools = n_whirlpools
        self.n_objects_per_whirlpool = n_objects_per_whirlpool
        self.n_pop = n_whirlpools + n_whirlpools * n_objects_per_whirlpool
        self.n_objects = self.n_pop - n_whirlpools
        self.max_iter = max_iter

        self.whirlpools = None
        self.best_solution = None
        self.best_value = float("inf")
        self.best_costs = np.zeros(max_iter)
        self.mean_costs = np.zeros(max_iter)

    def initialize_whirlpools(self):
        """Generate initial population and organize into whirlpools"""
        objects = []
        for _ in range(self.n_pop):
            position = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
            cost = self.objective_function(position)
            objects.append({"position": position, "cost": cost, "delta": 0})

        # Sort objects by cost
        objects = sorted(objects, key=lambda x: x["cost"])
        
        # Initialize whirlpools
        self.whirlpools = []
        for i in range(self.n_whirlpools):
            whirlpool = {
                "position": objects[i]["position"].copy(),
                "cost": objects[i]["cost"],
                "delta": objects[i]["delta"],
                "n_objects": self.n_objects_per_whirlpool,
                "objects": []
            }
            self.whirlpools.append(whirlpool)

        # Distribute remaining objects to whirlpools
        remaining_objects = objects[self.n_whirlpools:]
        np.random.shuffle(remaining_objects)
        obj_idx = 0
        for i in range(self.n_whirlpools):
            self.whirlpools[i]["objects"] = remaining_objects[obj_idx:obj_idx + self.n_objects_per_whirlpool]
            obj_idx += self.n_objects_per_whirlpool

    def effects_of_whirlpools(self, iter):
        """Implement pseudocodes 1-5: Update objects and whirlpool positions"""
        for i in range(self.n_whirlpools):
            for j in range(self.whirlpools[i]["n_objects"]):
                # Compute influence from other whirlpools
                if self.n_whirlpools > 1:
                    J = []
                    for t in range(self.n_whirlpools):
                        if t != i:
                            J.append(
                                (abs(self.whirlpools[t]["cost"]) ** 1) * 
                                (abs(np.sum(self.whirlpools[t]["position"]) - 
                                     np.sum(self.whirlpools[i]["objects"][j]["position"]))) ** 0.5
                            )
                    J = np.array(J)
                    min_idx = np.argmin(J)
                    max_idx = np.argmax(J)
                    
                    d = np.random.rand(self.dim) * (self.whirlpools[min_idx]["position"] - 
                                                   self.whirlpools[i]["objects"][j]["position"])
                    d2 = np.random.rand(self.dim) * (self.whirlpools[max_idx]["position"] - 
                                                    self.whirlpools[i]["objects"][j]["position"])
                else:
                    d = np.random.rand(self.dim) * (self.whirlpools[i]["position"] - 
                                                   self.whirlpools[i]["objects"][j]["position"])
                    d2 = np.zeros(self.dim)
                    min_idx = i

                # Update delta
                self.whirlpools[i]["objects"][j]["delta"] += np.random.rand() * np.random.rand() * np.pi
                eee = self.whirlpools[i]["objects"][j]["delta"]
                fr0 = np.cos(eee)
                fr10 = -np.sin(eee)

                # Compute new position
                x = ((fr0 * d) + (fr10 * d2)) * (1 + abs(fr0 * fr10))
                RR = self.whirlpools[i]["position"] - x
                RR = np.clip(RR, self.bounds[:, 0], self.bounds[:, 1])
                cost = self.objective_function(RR)

                # Update if better
                if cost <= self.whirlpools[i]["objects"][j]["cost"]:
                    self.whirlpools[i]["objects"][j]["cost"] = cost
                    self.whirlpools[i]["objects"][j]["position"] = RR

                # Pseudocode 3: Random jump
                FE_i = (abs(np.cos(eee) ** 2 * np.sin(eee) ** 2)) ** 2
                if np.random.rand() < FE_i:
                    k = np.random.randint(self.dim)
                    self.whirlpools[i]["objects"][j]["position"][k] = np.random.uniform(
                        self.bounds[k, 0], self.bounds[k, 1])
                    self.whirlpools[i]["objects"][j]["cost"] = self.objective_function(
                        self.whirlpools[i]["objects"][j]["position"])

        # Pseudocode 4: Update whirlpool positions
        J2 = [wp["cost"] for wp in self.whirlpools]
        best_whirlpool_idx = np.argmin(J2)
        best_position = self.whirlpools[best_whirlpool_idx]["position"]
        best_cost = J2[best_whirlpool_idx]

        for i in range(self.n_whirlpools):
            J = []
            for t in range(self.n_whirlpools):
                cost_diff = self.whirlpools[t]["cost"] * abs(
                    np.sum(self.whirlpools[t]["position"]) - np.sum(self.whirlpools[i]["position"]))
                J.append(float("inf") if t == i else cost_diff)
            
            min_idx = np.argmin(J)
            self.whirlpools[i]["delta"] += np.random.rand() * np.random.rand() * np.pi
            d = self.whirlpools[min_idx]["position"] - self.whirlpools[i]["position"]
            fr = abs(np.cos(self.whirlpools[i]["delta"]) + np.sin(self.whirlpools[i]["delta"]))
            x = fr * np.random.rand(self.dim) * d

            new_position = self.whirlpools[min_idx]["position"] - x
            new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
            new_cost = self.objective_function(new_position)

            # Pseudocode 5: Selection
            if new_cost <= self.whirlpools[i]["cost"]:
                self.whirlpools[i]["position"] = new_position
                self.whirlpools[i]["cost"] = new_cost

            if best_cost < self.whirlpools[best_whirlpool_idx]["cost"]:
                self.whirlpools[i]["position"] = best_position
                self.whirlpools[i]["cost"] = best_cost

    def pseudocode6(self):
        """Implement pseudocode 6: Update whirlpool with best object if better"""
        for i in range(self.n_whirlpools):
            costs = [obj["cost"] for obj in self.whirlpools[i]["objects"]]
            min_cost_idx = np.argmin(costs)
            min_cost = costs[min_cost_idx]

            if min_cost <= self.whirlpools[i]["cost"]:
                best_object = self.whirlpools[i]["objects"][min_cost_idx]
                self.whirlpools[i]["objects"][min_cost_idx]["position"] = self.whirlpools[i]["position"].copy()
                self.whirlpools[i]["objects"][min_cost_idx]["cost"] = self.whirlpools[i]["cost"]
                self.whirlpools[i]["position"] = best_object["position"].copy()
                self.whirlpools[i]["cost"] = best_object["cost"]

    def optimize(self):
        """Run the TFWO optimization algorithm"""
        self.initialize_whirlpools()
        
        for iter in range(self.max_iter):
            # Update whirlpools and objects
            self.effects_of_whirlpools(iter)
            self.pseudocode6()

            # Find best whirlpool
            whirlpool_costs = [wp["cost"] for wp in self.whirlpools]
            best_idx = np.argmin(whirlpool_costs)
            best_cost = whirlpool_costs[best_idx]
            best_position = self.whirlpools[best_idx]["position"].copy()

            # Update global best
            if best_cost < self.best_value:
                self.best_solution = best_position
                self.best_value = best_cost

            # Store iteration results
            self.best_costs[iter] = best_cost
            self.mean_costs[iter] = np.mean(whirlpool_costs)

            print(f"Iter {iter + 1}: Best Cost = {best_cost}")

        return self.best_solution, self.best_value, {"best_costs": self.best_costs, "mean_costs": self.mean_costs}

# Example usage
if __name__ == "__main__":
    def shifted_schwefel_1_2(x):
        """Shifted Schwefel's Problem 1.2 (CEC 2005)"""
        o = np.array([35.6267, -82.9123, -10.6423, -83.5815, 83.1552, 47.0480, -89.4359, -27.4219, 
                      76.1448, -39.0595, 48.8857, -3.9828, -71.9243, 64.1947, -47.7338, -5.9896, 
                      -26.2828, -59.1811, 14.6028, -85.4780, -50.4901, 0.9240, 32.3978, 30.2388, 
                      -85.0949, 60.1197, -36.2183, -8.5883, -5.1971, 81.5531][:len(x)])
        x = x - o
        z = 0
        for i in range(len(x)):
            z += np.sum(x[:i+1]) ** 2
        return z

    dim = 30
    bounds = [(-100, 100)] * dim
    optimizer = TurbulentFlowWaterOptimizer(shifted_schwefel_1_2, dim, bounds)
    best_solution, best_value, history = optimizer.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
