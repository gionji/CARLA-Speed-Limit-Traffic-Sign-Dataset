import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class NeuralNetworkOptimizer:
    def __init__(self, objective_function, parameters):
        self.objective_function = objective_function
        self.parameters = parameters

    def create_neural_network(self):
        model = tf.keras.Sequential([
            layers.Input(shape=(len(self.parameters['names']),)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)  # Output layer with 1 neuron for the score
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_neural_network(self, X, y, epochs=10):
        model = self.create_neural_network()
        model.fit(X, y, epochs=epochs, verbose=0)
        return model

    def run(self, num_iterations=10):
        # Generate random initial parameters
        initial_params = np.random.uniform(low=self.parameters['bounds'][:, 0],
                                           high=self.parameters['bounds'][:, 1],
                                           size=(num_iterations, len(self.parameters['names'])))

        X_train = []
        y_train = []

        # Evaluate initial parameters and collect data for training
        for params in initial_params:
            score = self.objective_function(dict(zip(self.parameters['names'], params)), 0)
            X_train.append(params)
            y_train.append(score)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train the neural network
        model = self.train_neural_network(X_train, y_train)

        # Perform optimization using the trained neural network
        best_params = []
        best_score = float('inf')

        for _ in range(num_iterations):
            # Generate random candidate parameters
            candidate_params = np.random.uniform(low=self.parameters['bounds'][:, 0],
                                                 high=self.parameters['bounds'][:, 1],
                                                 size=(1, len(self.parameters['names'])))

            # Predict the score for the candidate parameters
            candidate_score = model.predict(candidate_params)

            # If the candidate is better, update best_params and best_score
            if candidate_score < best_score:
                best_params = candidate_params.flatten()
                best_score = candidate_score

        # Print the best parameter combinations and corresponding score
        print("Best Parameters:")
        for param_name, param_value in zip(self.parameters['names'], best_params):
            print(f"{param_name}: {param_value}")
        print(f"Best Score: {best_score}")

        return best_params, best_score
