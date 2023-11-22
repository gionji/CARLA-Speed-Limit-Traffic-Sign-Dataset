
import numpy as np

class BinarySearchAgent:
    def __init__(self, evaluate_params, params):
        self.evaluate_params = evaluate_params
        self.param_names = params['names']
        self.param_bounds = params['bounds']

    def run(self, num_iterations):
        # Initialize the bounds for each parameter
        bounds_dict = dict(zip(self.param_names, self.param_bounds))

        print('bound dict ', bounds_dict)

        for iteration in range(num_iterations):
            # Initialize the best parameters dictionary
            best_params = {}

            # Iterate over each parameter
            for param_name in self.param_names:
                low, high = bounds_dict[param_name]

                # Binary search to find the midpoint
                mid = (low + high) / 2.0

                # Evaluate the fitness for the lower and upper bounds
                score_low = self.evaluate_params({param_name: low}, iteration)
                score_high = self.evaluate_params({param_name: high}, iteration)

                # Evaluate the fitness for the midpoint
                score_mid = self.evaluate_params({param_name: mid}, iteration)

                # Update the bounds based on the scores
                if score_low < score_mid < score_high:
                    bounds_dict[param_name] = (low, mid)
                elif score_low > score_mid > score_high:
                    bounds_dict[param_name] = (mid, high)
                else:
                    # If the scores are not in order, choose the bounds with the lower scores
                    if score_low < score_high:
                        bounds_dict[param_name] = (low, mid)
                    else:
                        bounds_dict[param_name] = (mid, high)

                # Add the best parameter to the dictionary
                best_params[param_name] = mid

            # Print the best values in each iteration
            print(f"Iteration {iteration + 1}: Best Parameters = {best_params}, Best Score = {score_mid}")

        # Return the best parameters found
        return best_params