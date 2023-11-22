import numpy as np

class PSOAgent:
    def __init__(self, evaluate_params, params, num_particles=10, inertia=0.5, personal_coeff=1.5, social_coeff=1.5):
        self.evaluate_params = evaluate_params
        self.param_names = params['names']
        self.param_bounds = params['bounds']
        self.num_particles = num_particles
        self.inertia = inertia
        self.personal_coeff = personal_coeff
        self.social_coeff = social_coeff

        # Initialize particles
        self.particles = np.random.uniform(low=self.param_bounds[:, 0], high=self.param_bounds[:, 1], size=(num_particles, len(self.param_names)))
        self.velocities = np.zeros_like(self.particles)
        self.best_positions = self.particles.copy()
        self.best_scores = np.full(num_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def update_particles(self, iteration):
        for i in range(self.num_particles):
            # Evaluate the current position
            current_score = self.evaluate_params(dict(zip(self.param_names, self.particles[i])), iteration)

            # Update personal best
            if current_score < self.best_scores[i]:
                self.best_scores[i] = current_score
                self.best_positions[i] = self.particles[i]

            # Update global best
            if current_score < self.global_best_score:
                self.global_best_score = current_score
                self.global_best_position = self.particles[i]

            # Update velocities and positions
            inertia_term = self.inertia * self.velocities[i]
            personal_term = self.personal_coeff * np.random.rand() * (self.best_positions[i] - self.particles[i])
            social_term = self.social_coeff * np.random.rand() * (self.global_best_position - self.particles[i])

            self.velocities[i] = inertia_term + personal_term + social_term
            self.particles[i] = self.particles[i] + self.velocities[i]

            # Ensure particles stay within bounds
            self.particles[i] = np.clip(self.particles[i], self.param_bounds[:, 0], self.param_bounds[:, 1])

    def run(self, num_iterations):
        for iteration in range(num_iterations):
            self.update_particles(iteration)

        # Find the best parameters among all particles
        best_particle_index = np.argmin(self.best_scores)
        best_params = dict(zip(self.param_names, self.best_positions[best_particle_index]))

        print(f"Best Parameters: {best_params}, Best Score: {self.best_scores[best_particle_index]}")

        return best_params