import numpy as np


def loc_to_rangeangle(robot_pose, landmark_loc):
    """Convert a landmark location to a range and bearing.

    Parameters
    ----------
    robot_pose : numpy array
        The robot pose as (x, y, gamma).
    landmark_loc : numpy array
        The landmark location as (x, y).

    Returns
    -------
    numpy array
        The range and bearing as (range, bearing).
    """
    if len(landmark_loc) == 0:
        return np.array([])
    if len(robot_pose) == 0:
        print("Warning: robot pose is empty in loc_to_rangeangle()")
        return np.array([])
    # Get the robot pose
    robot_x = robot_pose[0]
    robot_y = robot_pose[1]
    robot_gamma = robot_pose[2]
    # Get the landmark location
    landmark_x = landmark_loc[0]
    landmark_y = landmark_loc[1]
    # Calculate the range and bearing
    range = np.linalg.norm(landmark_loc - robot_pose[0:2])
    bearing = np.arctan2(landmark_y - robot_y, landmark_x - robot_x) - robot_gamma
    return np.array([range, bearing])


def rangeangle_to_loc(robot_pose, rangeangle):
    """Convert a range and bearing to a landmark location.

    Parameters
    ----------
    robot_pose : numpy array
        The robot pose as (x, y, gamma).
    landmark_polar : numpy array
        The range and bearing as (range, bearing).

    Returns
    -------
    numpy array
        The landmark location as (x, y).
    """
    # Get the robot pose
    robot_x = robot_pose[0]
    robot_y = robot_pose[1]
    robot_gamma = robot_pose[2] % (2 * np.pi)
    # Get the range and bearing
    range = rangeangle[0]
    bearing = rangeangle[1] % (2 * np.pi)
    # Calculate the landmark location
    landmark_x = robot_x + range * np.cos(bearing + robot_gamma)
    landmark_y = robot_y + range * np.sin(bearing + robot_gamma)
    return np.array([landmark_x, landmark_y])

def polar_to_cartesian(r, theta):
    'Converts 2D polar coordinates (range, angle) to return equivalent catesian (x, y offsets) '
    # Calculate Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
        
    return x, y

def wrapped_mean(angles): 
    return np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
