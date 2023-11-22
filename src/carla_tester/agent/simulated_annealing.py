import numpy as np

class SimulatedAnnealingAgent:
    def __init__(self, evaluate_params, params, initial_temperature=100.0, cooling_rate=0.95, iterations_per_temperature=10):
        self.evaluate_params = evaluate_params
        self.param_names = params['names']
        self.param_bounds = params['bounds']
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.iterations_per_temperature = iterations_per_temperature

    def generate_neighbor(self, current_params):
        # Generate a random neighbor within the parameter bounds
        neighbor_params = current_params.copy()
        random_index = np.random.randint(len(self.param_names))
        neighbor_params[random_index] = np.random.uniform(self.param_bounds[random_index, 0], self.param_bounds[random_index, 1])
        return neighbor_params

    def acceptance_probability(self, current_score, neighbor_score):
        # Probability of accepting a worse solution
        if neighbor_score < current_score:
            return 1.0
        return np.exp((current_score - neighbor_score) / self.temperature)

    def run(self, n_iterations_unused):
        current_params = np.random.uniform(low=self.param_bounds[:, 0], high=self.param_bounds[:, 1])
        current_score = self.evaluate_params(dict(zip(self.param_names, current_params)), iteration_n=0)

        best_params = current_params
        best_score = current_score

        for iteration in range(1, self.iterations_per_temperature + 1):
            neighbor_params = self.generate_neighbor(current_params)
            neighbor_score = self.evaluate_params(dict(zip(self.param_names, neighbor_params)), iteration_n=iteration)

            probability = self.acceptance_probability(current_score, neighbor_score)
            if np.random.rand() < probability:
                current_params = neighbor_params
                current_score = neighbor_score

            if current_score < best_score:
                best_params = current_params
                best_score = current_score

            print(f"Iteration: {iteration}, Temperature: {self.temperature}, Score: {current_score}, Best Score: {best_score}")

        self.temperature *= self.cooling_rate

        return best_params