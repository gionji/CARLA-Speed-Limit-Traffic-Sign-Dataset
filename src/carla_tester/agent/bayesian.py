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
            n_jobs=-1,
            return_train_score=False
        )

        opt.fit(None)  # Pass a dummy X, as it's not used in your example

        best_params = opt.best_params_
        best_score = opt.best_score_

        print(f"Best Parameters: {best_params}, Best Score: {best_score}")

        return best_params

    



