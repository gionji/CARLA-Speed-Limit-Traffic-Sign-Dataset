from itertools import product
import numpy as np

class GridSearch:
    def __init__(self, evaluate_params, params, bins=4):
        self.evaluate_params = evaluate_params
        self.params = params
        self.bins = bins
        self.param_combinations = self.generate_param_combinations()

    def generate_param_combinations(self):
        param_ranges = [np.linspace(min_val, max_val, self.bins) for min_val, max_val in self.params['bounds']]
        return list(product(*param_ranges))

    def run(self, unused):
        best_score = None
        best_params = None

        for params_values in self.param_combinations:
            current_params = dict(zip(self.params['names'], params_values))

            print('Current params: ', current_params)

            score = self.evaluate_params(current_params, iteration_n=0)  # Assuming iteration_n is not used in the evaluation

            if best_score is None or score < best_score:
                best_score = score
                best_params = current_params

        print(f"Best Parameters: {best_params}, Best Score: {best_score}")

        return best_params



class RandomGridSearch:
    def __init__(self, evaluate_params, params, num_samples=10):
        self.evaluate_params = evaluate_params
        self.params = params
        self.num_samples = num_samples
        self.param_combinations = self.generate_param_combinations()

    def generate_param_combinations(self):
        param_ranges = [(min_val, max_val) for min_val, max_val in self.params['bounds']]
        param_combinations = [self.generate_random_params(param_ranges) for _ in range(self.num_samples)]
        return param_combinations

    def generate_random_params(self, param_ranges):
        return [np.random.uniform(min_val, max_val) for min_val, max_val in param_ranges]

    def run(self, unused):
        best_score = None
        best_params = None

        for params_values in self.param_combinations:
            current_params = dict(zip(self.params['names'], params_values))

            print('Current params: ', current_params)

            score = self.evaluate_params(current_params, iteration_n=0)  # Assuming iteration_n is not used in the evaluation

            if best_score is None or score < best_score:
                best_score = score
                best_params = current_params

        print(f"Best Parameters: {best_params}, Best Score: {best_score}")

        return best_params
    

    import numpy as np


class RandomSearch:
    def __init__(self, evaluate_params, params, num_samples=10, bins=4):
        self.evaluate_params = evaluate_params
        self.params = params
        self.num_samples = num_samples
        self.bins = bins
        self.param_combinations = self.generate_param_combinations()

    def generate_param_combinations(self):
        param_ranges = [np.linspace(min_val, max_val, self.bins) for min_val, max_val in self.params['bounds']]
        param_combinations = [self.generate_random_params(param_ranges) for _ in range(self.num_samples)]
        return param_combinations

    def generate_random_params(self, param_ranges):
        return [np.random.choice(bins) for bins in param_ranges]

    def run(self, unused):
        best_score = None
        best_params = None

        for params_values in self.param_combinations:
            current_params = dict(zip(self.params['names'], params_values))

            print('Current params: ', current_params)

            score = self.evaluate_params(current_params, iteration_n=0)  # Assuming iteration_n is not used in the evaluation

            if best_score is None or score < best_score:
                best_score = score
                best_params = current_params

        print(f"Best Parameters: {best_params}, Best Score: {best_score}")

        return best_params
