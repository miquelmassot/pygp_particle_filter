import copy
import numpy as np

from sklearn.neighbors import KernelDensity

from .particle import Particle
from .observation import weight_observation
from .tools import rangeangle_to_loc


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
        self.observations_rb = None
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
        if self.observations_rb is None:
            self.observations_rb = np.array([new_observations_rb])
        else:
            self.observations_rb = np.append(
                self.observations_rb, [new_observations_rb], axis=0
            )
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
        if self.observations_rb is None:
            self.observations_rb = [new_observations_rb]
        else:
            self.observations_rb.append(new_observations_rb)
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

    @property
    def x(self):
        """Returns the x of the particles."""
        x = []
        if len(self.particles) == 0:
            return x
        for p in self.particles:
            x.append(p.x)
        return np.array(x)

    @property
    def y(self):
        """Returns the y of the particles."""
        y = []
        if len(self.particles) == 0:
            return y
        for p in self.particles:
            y.append(p.y)
        return np.array(y)

    @property
    def gamma(self):
        """Returns the gamma of the particles."""
        gamma = []
        if len(self.particles) == 0:
            return gamma
        for p in self.particles:
            gamma.append(p.gamma)
        return np.array(gamma)

    def kde_pose(self, sigma_resolution=0.1, sampling_resolution=1000):
        """Returns the pose of the KDE of the particles."""
        kde_poses = []
        for i in range(len(self.particles[0].path)):
            x = np.array([p.path[i][1] for p in self.particles]).squeeze()
            y = np.array([p.path[i][2] for p in self.particles]).squeeze()
            gamma = np.array([p.path[i][3] for p in self.particles]).squeeze()
            locations = np.vstack([x, y, gamma])
            kde = KernelDensity(kernel="gaussian", bandwidth=sigma_resolution).fit(
                locations.T, sample_weight=self.weights.T
            )
            state_range_x = np.linspace(
                min(x),
                max(x),
                num=sampling_resolution,
            )
            state_range_y = np.linspace(
                min(y),
                max(y),
                num=sampling_resolution,
            )
            state_range_gamma = np.linspace(
                min(gamma),
                max(gamma),
                num=sampling_resolution,
            )
            state_range = np.vstack([state_range_x, state_range_y, state_range_gamma])
            density = kde.score_samples(state_range.T)
            est_x, est_y, est_gamma = state_range.T[density.argmax()]
            res = np.array([est_x, est_y, est_gamma])
            kde_poses.append(res)
        return np.array(kde_poses)

    def kde_observations(self, sigma_resolution=0.1, sampling_resolution=1000):
        poses = self.kde_pose(sigma_resolution, sampling_resolution)
        kde_obs = []
        for obs, pose in zip(self.observations_rb, poses):
            obs_xy = rangeangle_to_loc(pose, obs)
            kde_obs.append(obs_xy)
        return np.array(kde_obs)
