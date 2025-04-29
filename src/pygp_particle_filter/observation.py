import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)


def weight_observation(
    past_observations_rb,
    new_observation_rb,
    robot_fov,
    robot_range,
    observation_std,
    matern_length_scale,
    gpr_sample_cap = 20,
):
    """Weight the new observation based on the past observations.

    Parameters
    ----------
    past_observations_rb : numpy array
        The past observations in the robot frame (range, bearing).
    new_observation_rb : numpy array
        The new observation in the robot frame (range, bearing).
    robot_pose : numpy array
        The robot pose as (x, y, gamma).
    robot_fov : float
        The field of view in radians.
    robot_range : float
        The maximum range in meters.
    observation_std : float
        The standard deviation of the observation noise.
    matern_length_scale : float
        The length scale of the Matern kernel.
    """
    # Do not run on an empty past observations array
    if past_observations_rb is None:
        return 1.0
    if len(past_observations_rb) == 0:
        return 1.0

    # filter the past observations and keep only the ones within fov and range
    past_observations_rb_in_range = past_observations_rb[
        past_observations_rb[:, 0] < robot_range
    ]
    past_observations_rb_in_fov = past_observations_rb_in_range[
        np.abs(past_observations_rb_in_range[:, 1]) < robot_fov / 2
    ]

    # Do not run on an empty array
    if len(past_observations_rb_in_fov) < 5:
        return 1.0

    # Train the GP model with the past observations. Note that X and Y are swapped.
    obs_bearing = np.array(past_observations_rb_in_fov)[:, 1].reshape(-1, 1)
    obs_range = np.array(past_observations_rb_in_fov)[:, 0].reshape(-1, 1)

    # restrict the number of samples to use to train the GPR
    i = list(range(len(obs_bearing)))
        
    if len(obs_bearing) > gpr_sample_cap:
        j = np.random.choice(i,size = gpr_sample_cap, replace=False)
        obs_bearing = obs_bearing[j]
        obs_range = obs_range[j]

    mean_obs_range = np.mean(obs_range)
    kernel = (
        ConstantKernel(mean_obs_range)
        + Matern(
            length_scale=matern_length_scale, nu=3 / 2, length_scale_bounds="fixed"
        )
        + WhiteKernel(noise_level=observation_std**2)
    )
    gp = gaussian_process.GaussianProcessRegressor(
        kernel=kernel, alpha=observation_std**2, n_restarts_optimizer=0
    )
    gp.fit(obs_bearing, obs_range)

    # Predict the new observation. Note that X and Y are swapped.
    new_obs_bearing = np.array(new_observation_rb[:, 1]).reshape(-1, 1)
    new_obs_range = np.array(new_observation_rb[:, 0])
    mean_range_prediction, std_range_prediction = gp.predict(
        new_obs_bearing, return_std=True
    )

    sum_squared_stds = (
        std_range_prediction**2
        + (observation_std * np.ones(std_range_prediction.shape)) ** 2
    )

    # Calculate the weight
    weight = np.exp(
        -1 * (mean_range_prediction - new_obs_range) ** 2 / sum_squared_stds
    ) / np.sqrt(2 * np.pi * sum_squared_stds)
    # Normalise the weight
    weight /= len(weight)  # HACK
    mean_weight = np.mean(weight)
    return mean_weight
