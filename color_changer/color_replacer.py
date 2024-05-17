import numpy as np


def color_replacer(im1, im2):
    """
    Return a color replacer of im1 by im2 mask .

    Parameters
    ----------
    im1 : array_like
        Input array
    im2 : array_like
        Input array


    Returns
    -------
    im : ndarray
        Array of the same type and shape as `im1` and `im2`.


    """
    return np.take_along_axis(
            np.sort(im2, axis=1), 
            np.transpose(np.tile(np.argsort(np.argsort(im1.mean(axis=2), axis=1), axis=1), (3, 1, 1)), (1,2,0)), 
            axis=1
        )