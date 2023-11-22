import numpy as np
from cma import CMAEvolutionStrategy

class CMAESAgent:
    def __init__(self, evaluate_params, params, population_size=10, sigma_init=0.5, iterations=100):
        self.evaluate_params = evaluate_params
        self.param_names = params['names']
        self.param_bounds = params['bounds']
        self.population_size = population_size
        self.sigma_init = sigma_init
        self.iterations = iterations
        self.current

    def run(self, n_iter_unused):
        def objective_function(x):
            params_dict = dict(zip(self.param_names, x))
            return -self.evaluate_params(params_dict, iteration_n=0)

        # Initial guess for the parameters
        initial_params = np.random.uniform(low=self.param_bounds[:, 0], high=self.param_bounds[:, 1])

        # Run CMA-ES optimization
        es = CMAEvolutionStrategy(initial_params, self.sigma_init, {'popsize': self.population_size})
        best_params, _ = es.optimize(objective_function, iterations=self.iterations)

        return best_params
