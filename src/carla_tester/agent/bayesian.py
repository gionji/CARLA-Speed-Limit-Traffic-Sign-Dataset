from skopt import BayesSearchCV
from sklearn.base import BaseEstimator
import numpy as np

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
            n_jobs=1,
            return_train_score=False
        )

        opt.fit(None)  # Pass a dummy X, as it's not used in your example

        best_params = opt.best_params_
        best_score = opt.best_score_

        print(f"Best Parameters: {best_params}, Best Score: {best_score}")

        return best_params



from scipy.stats import norm

class GaussianProcess:
    def __init__(self, kernel, noise=1e-4):
        self.kernel = kernel
        self.noise = noise

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self.kernel(X, X) + np.eye(len(X)) * self.noise
        self.L = np.linalg.cholesky(K)

    def predict(self, X):
        K_s = self.kernel(self.X_train, X)
        Lk = np.linalg.solve(self.L, K_s)
        mu = np.dot(Lk.T, np.linalg.solve(self.L, self.y_train))
        K_ss = self.kernel(X, X)
        sigma = np.sqrt(np.diag(K_ss) - np.sum(Lk**2, axis=0))
        return mu, sigma

def squared_exponential_kernel(x1, x2, length_scale=1.0, noise=1e-4):
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return np.exp(-0.5 / length_scale**2 * sqdist) + np.eye(len(x1)) * noise

def expected_improvement(mu, sigma, best_observed_value, xi=0.01):
    Z = (mu - best_observed_value - xi) / sigma
    return (mu - best_observed_value - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

class BayesianOptimizer:
    def __init__(self, objective_function, params, num_iterations):
        self.objective_function = objective_function
        self.params = params
        self.num_iterations = num_iterations
        self.bounds = np.array(params['bounds'])
        self.kernel = squared_exponential_kernel
        self.gp = None

    def run(self, unused):
        # Randomly sample initial points
        X_observed = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(5, len(self.bounds)))
        y_observed = np.array([self.objective_function(params, 0) for params in X_observed])

        # GP regression
        self.gp = GaussianProcess(self.kernel)
        self.gp.fit(X_observed, y_observed)

        for i in range(self.num_iterations):
            # Optimize acquisition function to get the next point to sample
            x_next = self.optimize_acquisition()
            
            # Evaluate the true objective function at the new point
            y_next = self.objective_function(x_next, i)
            
            # Add the new observation to the dataset
            X_observed = np.vstack((X_observed, x_next))
            y_observed = np.vstack((y_observed, y_next))
            
            # Update the GP regression model
            self.gp.fit(X_observed, y_observed)

        best_params = X_observed[np.argmin(y_observed)]
        best_score = np.min(y_observed)

        print(f"Best Parameters: {best_params}, Best Score: {best_score}")

        return best_params

    def optimize_acquisition(self):
        x_random = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        y_random = expected_improvement(self.gp.predict(x_random.reshape(1, -1))[0][0],
                                        self.gp.predict(x_random.reshape(1, -1))[1][0],
                                        np.min(self.gp.y_train))
        return x_random
    



import numpy as np

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

class BayesianOptimizer03:
    def __init__(self, objective_function, params, num_iterations, acquisition_function='expected_improvement', xi=0.01):
        self.objective_function = objective_function
        self.params = params
        self.num_iterations = num_iterations
        self.acquisition_function = acquisition_function
        self.xi = xi
        self.bounds = np.array(params['bounds'])
        self.best_observed_value = np.inf
        self.X_observed = []
        self.y_observed = []

    def expected_improvement(self, x, gp):
        mu, sigma = gp.predict(np.array(x).reshape(1, -1), return_std=True)
        sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero
        Z = (mu - self.best_observed_value - self.xi) / sigma
        return (mu - self.best_observed_value - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

    def optimize_acquisition(self, gp):
        x_random = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        y_random = self.expected_improvement(x_random, gp)
        return x_random

    def run(self, uuu):
        # Initialize the GP model
        gp = GaussianProcessRegressor(kernel=ConstantKernel() * Matern(length_scale=1.0), n_restarts_optimizer=10)

        for i in range(self.num_iterations):
            # Optimize acquisition function to get the next point to sample
            x_next = self.optimize_acquisition(gp)
            
            # Convert the array to a dictionary with parameter names
            x_next_dict = {'names': self.params['names'], 'values': x_next.tolist()}
            
            # Evaluate the true objective function at the new point
            y_next = self.objective_function(x_next_dict, i)
            
            # Update the best observed value
            self.best_observed_value = min(self.best_observed_value, y_next)
            
            # Add the new observation to the dataset
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)
            
            # Update the GP regression model
            gp.fit(np.array(self.X_observed), np.array(self.y_observed))

        best_params_index = np.argmin(self.y_observed)
        best_params = self.X_observed[best_params_index]
        best_score = self.y_observed[best_params_index]

        print(f"Best Parameters: {best_params}, Best Score: {best_score}")

        return best_params