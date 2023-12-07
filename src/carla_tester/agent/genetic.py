import random
import numpy as np

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
        Select parents based on lower fitness scores.
        """
        parents_indices = np.argsort(fitness_scores)[:2]  # Select the bottom 2 individuals
        parent1, parent2 = self.population[parents_indices[0]], self.population[parents_indices[1]]
        return parent1, parent2, parents_indices[0]

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
            fitness_scores = [self.evaluate_params(dict(zip(self.params['names'], params)), iteration) for params in self.population]

            # Select parents based on lower fitness scores
            parent1, parent2, best_index = self.select_parents(fitness_scores)

            # Crossover and mutation to create a new generation
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, mutation_rate=0.1)

            # Replace the best individual with the new child only if the child has a lower fitness score
            child_fitness = self.evaluate_params(dict(zip(self.params['names'], child)), iteration)
            if child_fitness < fitness_scores[best_index]:
                self.population[best_index] = child

            # Print the best individual in each iteration
            best_index = np.argmin(fitness_scores)
            best_params = self.population[best_index]
            best_score = fitness_scores[best_index]
            print(f"Iteration {iteration + 1}: Best Parameters = {best_params}, Best Score = {best_score}")

        # Return the best parameters found
        best_index = np.argmin(fitness_scores)

        return self.population[best_index]
