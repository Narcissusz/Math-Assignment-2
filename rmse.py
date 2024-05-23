import numpy as np

def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)
    # Compute the squared differences
    squared_diff = (pred - tar) ** 2
    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)
    # Take the square root of the mean squared differences to get RMSE
    rmse = np.sqrt(mean_squared_diff)
    return rmse