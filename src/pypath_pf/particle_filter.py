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

    def add_observation(self, range, bearing):
        for p in self.particles:
            p.add_observation(range, bearing)

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

    def observation_update(self, lidar_observations, observation_std, length_scale):
        """
        Update particle weights based on lidar observations.

        Input:
            lidar_observations: list of [range, bearing] observations
        """
        for particle in self.particles:
            particle.weight *= np.mean(
                weight_observation(
                    particle.observations,
                    lidar_observations,
                    particle.x,
                    particle.y,
                    particle.gamma,
                    particle.fov,
                    particle.range,
                    observation_std,
                    length_scale,
                )
            )
        self.resampling()

    def data_association(self, particle, range, bearing):
        """
        For a particle, compute likelihood of correspondence for all observations.
        Choose observation according to ML (Maximum Likelihood).

        returns the observation index of the ML observation
        """
        # TODO
        return 0

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
