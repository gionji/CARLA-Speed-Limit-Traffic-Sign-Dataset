

class SimulationParameter:
    def __init__(self, category, name, range, n_bins, discrete_values=None):
        self.category = category
        self.name = name 
        self.range = range 
        self.n_bins = n_bins
        self.discrete = discrete_values