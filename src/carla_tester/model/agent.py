import carla
import random
import numpy as np


class RandomStupidAgent:
    def __init__(self, experiment):
        self.experiment = experiment
        self.current_parameters_valuess = None
        self.parameters = None

    def set_parameters(self, params):
        self.parameters = params

    def perform_action(self, score):
        # calculate new values
        weather_params = self.generate_random_weather_parameters( self.parameters )
        # apply
        self.current_parameters_valuess = weather_params
        
        return self.current_parameters_valuess  

    def get_parameter_value(self, min_value, max_value, num_bins):
        # Calculate the width of each bin
        bin_width = (max_value - min_value) / num_bins
        # Generate random value from one of the bins
        sampled_value = min_value + (random.choice(range(num_bins + 1)) * bin_width)
        return sampled_value

    def generate_random_weather_parameters(self, parameter_names, n_bins=8):
        weather_parameters = carla.WeatherParameters()
        for param_name in parameter_names:
            if hasattr(carla.WeatherParameters, param_name):
                if param_name == 'sun_azimuth_angle':
                    random_value = self.get_parameter_value(0, 360, n_bins)
                elif param_name == 'sun_altitude_angle':
                    random_value = self.get_parameter_value(-45, 90, n_bins)
                elif param_name == 'cloudiness':
                    random_value = self.get_parameter_value(40, 90, n_bins)
                else:
                    random_value = self.get_parameter_value(0, 100, n_bins)

                setattr(weather_parameters, param_name, random_value)
                #print("Randomized param", param_name, random_value)
        return weather_parameters
    





class GeneticAlgorithm:
    def __init__(self, evaluate_params, population_size, params):
        self.evaluate_params = evaluate_params
        self.population_size = population_size
        self.params = params
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Initialize the population with random parameters within specified bounds.
        """
        population = []
        for _ in range(self.population_size):
            params = np.random.uniform(self.params['bounds'][:, 0], self.params['bounds'][:, 1])
            population.append(params)
        return population

    def select_parents(self, fitness_scores):
        """
        Select parents based on fitness scores.
        """
        parents_indices = np.argsort(fitness_scores)[:2]  # Select the top 2 individuals
        parent1, parent2 = self.population[parents_indices[0]], self.population[parents_indices[1]]
        return parent1, parent2

    def crossover(self, parent1, parent2):
        """
        Perform crossover operation to generate a new individual.
        """
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def mutate(self, individual, mutation_rate):
        """
        Perform mutation operation on an individual.
        """
        mutation_mask = np.random.rand(len(individual)) < mutation_rate
        individual += mutation_mask * np.random.uniform(-0.5, 0.5)
        # Ensure mutated values are within bounds
        individual = np.clip(individual, self.params['bounds'][:, 0], self.params['bounds'][:, 1])
        return individual

    def run(self, num_iterations):
        """
        Run the genetic algorithm for a specified number of iterations.
        """
        for iteration in range(num_iterations):
            # Evaluate the fitness of each individual in the population
            fitness_scores = [self.evaluate_params(params, iteration) for params in self.population]

            # Select parents based on fitness scores
            parent1, parent2 = self.select_parents(fitness_scores)

            # Crossover and mutation to create a new generation
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, mutation_rate=0.1)

            # Replace the best individual with the new child
            best_index = np.argmax(fitness_scores)
            self.population[best_index] = child

            # Print the best individual in each iteration
            best_index = np.argmax(fitness_scores)
            best_params = self.population[best_index]
            best_score = fitness_scores[best_index]
            print(f"Iteration {iteration + 1}: Best Parameters = {best_params}, Best Score = {best_score}")

        # Return the best parameters found
        best_index = np.argmax(fitness_scores)

        return self.population[best_index]
    


from skopt import BayesSearchCV
from sklearn.base import BaseEstimator

class EvaluationWrapper(BaseEstimator):
    def __init__(self, experiment_instance):
        self.experiment_instance = experiment_instance

    def fit(self, X, y=None):
        # This method is called by BayesSearchCV, but we don't need to train anything
        return self

    def score(self, params):
        # Evaluate and return the score using the provided parameters
        return -self.experiment_instance.evaluate_params(params, iteration_n=0)


class BayesianOptimization:
    def __init__(self, experiment_instance, params):
        self.experiment_instance = experiment_instance
        self.params = params

    def run(self, num_iterations):
        search_space = {name: (low, high) for name, (low, high) in zip(self.params['names'], self.params['bounds'])}

        # Wrap the evaluation method using the EvaluationWrapper
        evaluation_wrapper = EvaluationWrapper(self.experiment_instance)

        opt = BayesSearchCV(
            evaluation_wrapper,
            search_space,
            n_iter=num_iterations,
            random_state=42,
            n_jobs=-1,
            return_train_score=False
        )

        opt.fit(None)  # Pass a dummy X, as it's not used in your example

        best_params = opt.best_params_
        best_score = opt.best_score_

        print(f"Best Parameters: {best_params}, Best Score: {best_score}")

        return best_params

    



