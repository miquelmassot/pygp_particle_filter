import numpy as np
from .tools import loc_to_rangeangle, rangeangle_to_loc


class Particle:
    __slots__ = (
        "timestamp",
        "x",
        "y",
        "gamma",
        "fov",
        "range",
        "weight",
        "observations",
        "path",
        "motion_noise",
    )

    def __init__(
        self,
        t=0.0,
        x=0.0,
        y=0.0,
        gamma=0.0,
        num_particles=1,
        motion_noise=[0.01, 0.01, 0.01, 0.01, 0.01],
        fov=np.pi * 2 / 3,
        range=2.0,
    ):
        """_summary_

        Parameters
        ----------
        t : float, optional
            Starting timestamp in seconds, by default 0.0
        x : float, optional
            Starting x position in meters, by default 0.0
        y : float, optional
            Starting y position in meters, by default 0.0
        gamma : float, optional
            Starting orientation in radians, by default 0.0
        num_particles : int, optional
            The number of particles, by default 1
        motion_noise : list, optional
            Motion model noise as a list for x, y, gamma, x dot, gamma dot, by default [0.01, 0.01, 0.01, 0.01, 0.01]
        """
        # Robot state: [timestamp, x, y, gamma]
        self.timestamp = t
        self.x = x
        self.y = y
        self.gamma = gamma % (2 * np.pi)
        self.fov = fov
        self.range = range
        # Weight
        self.weight = 1.0 / num_particles
        # Observations
        self.observations = None
        self.path = None

        # Noises
        # motion_noise: [noise_x, noise_y, noise_gamma, noise_v, noise_w]
        # (in meters or rad).
        self.motion_noise = motion_noise

        # Apply Gaussian noise to the robot state
        self.initialise()

    def add_observations(self, new_observations_rb):
        """Adds new observations to the particle

        Parameters
        ----------
        new_observations_rb : np.ndarray
            Array containing (range, bearing) for all measurements.
        """
        for obs in new_observations_rb:
            obs_xy = rangeangle_to_loc(self.pose, obs)
            if self.observations is None:
                self.observations = np.array([obs_xy])
            else:
                self.observations = np.append(self.observations, [obs_xy], axis=0)

    def initialise(self):
        # Apply Gaussian noise to the robot state
        self.x = np.random.normal(self.x, self.motion_noise[0])
        self.y = np.random.normal(self.y, self.motion_noise[1])
        self.gamma = np.random.normal(self.gamma, self.motion_noise[2]) % (2 * np.pi)
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
        self.x += v * np.cos(self.gamma) * delta_t
        self.y += v * np.sin(self.gamma) * delta_t
        self.gamma += w * delta_t

        robot_path = np.array([[self.timestamp, self.x, self.y, self.gamma]])

        if self.path is None:
            self.path = robot_path
        else:
            self.path = np.append(self.path, robot_path, axis=1)

        # Limit θ within [0, 2*np.pi]
        self.gamma = self.gamma % (2 * np.pi)

    @property
    def pose(self):
        return np.array([self.x, self.y, self.gamma])

    @pose.setter
    def pose(self, pose):
        self.x = pose[0]
        self.y = pose[1]
        self.gamma = pose[2] % (2 * np.pi)

    @property
    def observations_rangeangle(self):
        """Returns past observations in the robot frame as (range, bearing)."""
        if self.observations is None:
            return None
        rangeangles = []
        for obs in self.observations:
            rangeangles.append(loc_to_rangeangle(self.pose, obs))
        rangeangles = np.array(rangeangles)
        return rangeangles
