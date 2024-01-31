import copy
import numpy as np

from .particle import Particle
from .tools import weight_observation


class ParticleFilter:
    def __init__(self, num_particles, motion_noise=[0.01, 0.01, 0.01, 0.01, 0.01]):
        self.num_particles = num_particles
        self.particles = []
        for _ in range(num_particles):
            p = Particle(num_particles=num_particles, motion_noise=motion_noise)
            self.particles.append(p)

    def predict(self, control):
        for p in self.particles:
            p.predict(control)

    def add_observations(self, new_observations_rb):
        """Adds new observations to the particle filter.

        Parameters
        ----------
        new_observations_rb : np.ndarray
            Array containing (range, bearing) for all measurements.
        """
        for p in self.particles:
            p.add_observations(new_observations_rb)

    def weights_normalisation(self):
        sum = 0.0
        for p in self.particles:
            sum += p.weight
        num_p = len(self.particles)
        if sum < 1e-10:
            self.weights = [1.0 / num_p] * num_p
        self.weights /= sum

    def importance_sampling(self):
        new_indexes = np.random.choice(
            len(self.particles), len(self.particles), replace=True, p=self.weights
        )
        new_particles = []
        for index in new_indexes:
            new_particles.append(copy.deepcopy(self.particles[index]))
        self.particles = new_particles

    def number_effective_particles(self):
        sum = 0.0
        for p in self.particles:
            sum += p.weight**2
        return 1.0 / sum

    def resampling(self):
        if self.number_effective_particles() < self.num_particles / 2:
            self.importance_sampling()
        else:
            self.weights_normalisation()

    def observation_update(self, new_observations_rb, observation_std, length_scale):
        """
        Update particle weights based on lidar observations.

        Input:
            lidar_observations: list of [range, bearing] observations
        """
        for particle in self.particles:
            particle.weight *= weight_observation(
                particle.observations_rangeangle,
                new_observations_rb,
                particle.fov,
                particle.range,
                observation_std,
                length_scale,
            )
            particle.add_observations(new_observations_rb)
        self.resampling()

    @property
    def weights(self):
        w = []
        if len(self.particles) == 0:
            return w
        for p in self.particles:
            w.append(p.weight)
        return w

    @weights.setter
    def weights(self, new_weights):
        for i in range(len(self.particles)):
            self.particles[i].weight = new_weights[i]
