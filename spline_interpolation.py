import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_with_cubic_splines(points):
    x = points[:, 0]
    y = points[:, 1]

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_indices]

    cs = CubicSpline(unique_x, unique_y)

    return cs