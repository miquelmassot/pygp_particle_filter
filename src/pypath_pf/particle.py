import numpy as np
from .tools import loc_to_rangeangle, rangeangle_to_loc


class Particle:
    __slots__ = ("x", "y", "gamma", "weight", "observations", "path", "motion_noise")

    def __init__(
        self,
        t0=0.0,
        x0=0.0,
        y0=0.0,
        gamma0=0.0,
        num_particles=1,
        motion_noise=[0.01, 0.01, 0.01, 0.01, 0.01],
        fov=np.pi * 2 / 3,
        range=2.0,
    ):
        """_summary_

        Parameters
        ----------
        t0 : float, optional
            Starting timestamp in seconds, by default 0.0
        x0 : float, optional
            Starting x position in meters, by default 0.0
        y0 : float, optional
            Starting y position in meters, by default 0.0
        gamma0 : float, optional
            Starting orientation in radians, by default 0.0
        num_particles : int, optional
            The number of particles, by default 1
        motion_noise : list, optional
            Motion model noise as a list for x, y, gamma, x dot, gamma dot, by default [0.01, 0.01, 0.01, 0.01, 0.01]
        """
        # Robot state: [timestamp, x, y, gamma]
        self.timestamp = t0
        self.x = x0
        self.y = y0
        self.gamma = gamma0
        self.fov = fov
        self.range = range
        # Weight
        self.weight = 1.0 / num_particles
        # Observations
        self.observations = np.array([[]])
        self.path = np.array([[]])

        # Noises
        # motion_noise: [noise_x, noise_y, noise_theta, noise_v, noise_w]
        # (in meters or rad).
        self.motion_noise = motion_noise

    def add_observations(self, new_observations_rb):
        """Adds new observations to the particle

        Parameters
        ----------
        new_observations_rb : np.ndarray
            Array containing (range, bearing) for all measurements.
        """
        for obs in new_observations_rb:
            self.observations.append(rangeangle_to_loc(self.pose, obs))

    def initialise(self):
        # Apply Gaussian noise to the robot state
        self.x = np.random.normal(self.x, self.motion_noise[0])
        self.y = np.random.normal(self.y, self.motion_noise[1])
        self.gamma = np.random.normal(self.theta, self.motion_noise[2])
        self.path = np.array([[self.timestamp, self.x, self.y, self.gamma]])

    def predict(self, control):
        """
        Sample next state X_t from current state X_t-1 and control U_t with
        added motion noise.

        Input:
            control: control input U_t.
                     [timestamp, v_t, w_t]
        """
        # Apply Gaussian noise to control input
        v = np.random.normal(control[1], self.motion_noise[3])
        w = np.random.normal(control[2], self.motion_noise[4])

        delta_t = control[0] - self.timestamp

        # Compute updated [timestamp, x, y, gamma]
        self.timestamp = control[0]
        self.x += (control[1] + v) * np.cos(self.gamma) * delta_t
        self.y += (control[1] + v) * np.sin(self.gamma) * delta_t
        self.gamma += (control[2] + w) * delta_t

        self.path.append(np.array([[self.timestamp, self.x, self.y, self.gamma]]))

        # Limit θ within [-pi, pi]
        if self.gamma > np.pi:
            self.gamma -= 2 * np.pi
        elif self.gamma < -np.pi:
            self.gamma += 2 * np.pi

    @property
    def pose(self):
        return np.array([self.x, self.y, self.gamma])

    @pose.setter
    def pose(self, pose):
        self.x = pose[0]
        self.y = pose[1]
        self.gamma = pose[2]

    @property
    def observations_rangeangle(self):
        """Returns past observations in the robot frame as (range, bearing)."""
        rangeangles = []
        for obs in self.observations:
            rangeangles.append(loc_to_rangeangle(self.pose, obs))
        return rangeangles
