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
    