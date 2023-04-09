import numpy as np

class WhaleOptimization:
    def __init__(self, search_space, cost_function, n_agents=5, a_max=2, a_min=0, b_max=1, b_min=0, max_iter=100):
        """
        Whale Optimization Algorithm.

        Parameters:
        search_space (list): Search space of the problem.
        cost_function (function): Cost function of the problem.
        n_agents (int): Number of agents (default=5).
        a_max (int): Maximum value of a parameter (default=2).
        a_min (int): Minimum value of a parameter (default=0).
        b_max (int): Maximum value of b parameter (default=1).
        b_min (int): Minimum value of b parameter (default=0).
        max_iter (int): Maximum number of iterations (default=100).
        """
        self.search_space = search_space
        self.cost_function = cost_function
        self.n_agents = n_agents
        self.a_max = a_max
        self.a_min = a_min
        self.b_max = b_max
        self.b_min = b_min
        self.max_iter = max_iter

    def _initialize_agents(self):
        """
        Initialize the agents.
        """
        agents = []
        for i in range(self.n_agents):
            agent = {
                "position": np.random.uniform(self.search_space[0], self.search_space[1], size=len(self.search_space)),
                "cost": float("inf")
            }
            agents.append(agent)
        return agents

    def _update_a_b(self, iter):
        """
        Update the a and b parameters.

        Parameters:
        iter (int): Current iteration.
        """
        self.a = self.a_max - ((self.a_max - self.a_min) / self.max_iter) * iter
        self.b = self.b_min + ((self.b_max - self.b_min) / self.max_iter) * iter

    def _update_position(self, x, x_neigh_best, x_global_best):
        """
        Update the position of the agent.

        Parameters:
        x (np.array): Current position of the agent.
        x_neigh_best (np.array): Best position of the agent's neighbourhood.
        x_global_best (np.array): Global best position.
        """
        r1, r2 = np.random.uniform(size=2)
        a = 2 * self.a * r1 - self.a
        c = 2 * r2
        distance_to_neigh_best = np.abs(x_neigh_best - x)
        x1 = x_neigh_best - a * distance_to_neigh_best
        distance_to_global_best = np.abs(x_global_best - x)
        x2 = x_global_best - a * distance_to_global_best
        new_x = (x1 + x2) / 2 - c * distance_to_neigh_best
        return new_x

    def optimize(self):
        """
        Optimize using Whale Optimization Algorithm.
        """
        agents = self._initialize_agents()
        global_best = {"position": None, "cost": float("inf")}
        neighbourhood_size = int(np.ceil(self.n_agents / 2))
        for i in range(self.max_iter):
            for j, agent in enumerate(agents):
                neighbours_idx = np.random.choice([idx for idx in range(self.n_agents) if idx != j], size=neighbourhood_size, replace=False)
                neighbours = [agents[n_idx] for n_idx in neighbours_idx]
                neighbours_best = min(neighbours, key=lambda n: n["cost"])
                x_neigh_best = neighbours_best["position"]
            x_global_best = global_best["position"]
            new_x = self._update_position(agent["position"], x_neigh_best, x_global_best)
            new_x = np.clip(new_x, self.search_space[0], self.search_space[1])
            new_cost = self.cost_function(new_x)
            if new_cost < agent["cost"]:
                agent["position"] = new_x
                agent["cost"] = new_cost
                if new_cost < global_best["cost"]:
                    global_best["position"] = new_x
                    global_best["cost"] = new_cost
        self._update_a_b(i)
    return global_best["position"], global_best["cost"]


