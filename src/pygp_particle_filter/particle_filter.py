import copy
import numpy as np

from .particle import Particle
from .observation import weight_observation


class ParticleFilter:
    def __init__(
        self,
        num_particles,
        x_range=[-1.0, 1.0],
        y_range=[-1.0, 1.0],
        gamma_range=[-np.pi, np.pi],
        motion_noise=[0.01, 0.01, 0.01, 0.01, 0.01],
    ):
        """Constructor for the particle filter.

        Parameters
        ----------
        num_particles : int
            Number of particles.
        x_range : list, optional
            Range of x values, by default [-1.0, 1.0]
        y_range : list, optional
            Range of y values, by default [-1.0, 1.0]
        gamma_range : list, optional
            Range of gamma values, by default [-np.pi, np.pi]
        motion_noise : list, optional
            Motion noise as (x, y, gamma, v, w), by default [0.01, 0.01, 0.01, 0.01, 0.01]
        """
        self.num_particles = num_particles
        self.particles = []
        for _ in range(num_particles):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            gamma = np.random.uniform(gamma_range[0], gamma_range[1])
            p = Particle(
                x=x,
                y=y,
                gamma=gamma,
                num_particles=num_particles,
                motion_noise=motion_noise,
            )
            self.particles.append(p)

    def predict(self, control):
        """Prediction step of the particle filter.

        Parameters
        ----------
        control: np.ndarray
            control input U_t as [timestamp, v_t, w_t]
        """
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
        """Normalise the particle weights so that they sum to 1.0."""
        sum = 0.0
        for p in self.particles:
            sum += p.weight
        num_p = len(self.particles)
        if sum < 1e-10:
            self.weights = [1.0 / num_p] * num_p
        self.weights /= sum

    def importance_sampling(self):
        """Perform importance sampling."""
        new_indexes = np.random.choice(
            len(self.particles), len(self.particles), replace=True, p=self.weights
        )
        new_particles = []
        for index in new_indexes:
            new_particles.append(copy.deepcopy(self.particles[index]))
        self.particles = new_particles

    def number_effective_particles(self):
        """Calculate the number of effective particles."""
        sum = 0.0
        for p in self.particles:
            sum += p.weight**2
        return 1.0 / sum

    def resampling(self):
        """Resampling step of the particle filter only if the number of effective particles is less than half of the total number of particles."""
        print("Number of effective particles: ", self.number_effective_particles())
        if self.number_effective_particles() < self.num_particles / 2:
            print("Resampling")
            self.importance_sampling()
        self.weights_normalisation()

    def observation_update(self, new_observations_rb, observation_std, length_scale):
        """
        Update particle weights based on lidar observations.

        Input:
            lidar_observations: list of [range, bearing] observations
        """
        if len(new_observations_rb) == 0:
            return
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
        self.weights_normalisation()
        self.resampling()

    @property
    def weights(self):
        """Returns the weights of the particles."""
        w = []
        if len(self.particles) == 0:
            return w
        for p in self.particles:
            w.append(p.weight)
        w = np.array(w)
        return w

    @weights.setter
    def weights(self, new_weights):
        for i in range(len(self.particles)):
            self.particles[i].weight = new_weights[i]

    @property
    def mean_pose(self):
        """Returns the mean state of the particles."""
        mean_pose = np.zeros(3)
        if len(self.particles) == 0:
            return mean_pose
        for p in self.particles:
            mean_pose += p.weight * p.pose
        return mean_pose
