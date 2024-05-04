import numpy as np
from scipy import ndimage

def im_2d_sort(im, axis=[0]):
    """
    Return a sorted copy of an image array.

    Parameters
    ----------
    im : array_like
        Input array
    axes : [0, 1, 2], tuple or list of ints, optional

    Returns
    -------
    sorted_im : ndarray
        Array of the same type and shape as `im`.


    """
    im_tmp = im

    for ax in axis:
        im_tmp = np.sort(im_tmp, axis=ax)

    return im_tmp

def im_1d_sort(im, axis=(0, 1)):
    """
    Return a 1 line sorted copy of an image array.

    Parameters
    ----------
    im : array_like
        Input array
    axes : {(0, 1), (1, 0)}, tuple or list of ints, optional

    Returns
    -------
    sorted_im : ndarray
        Array of the same type and shape as `im`.

    """
    im_tmp = np.transpose(im, axis+(2,))
    im_shape = im_tmp.shape
    im_tmp = im_tmp.reshape(-1, 3)
    im_tmp = np.sort(im_tmp, axis=0)

    if axis == (0, 1):
        return im_tmp.reshape(im_shape)
    else:
        return np.transpose(im_tmp.reshape(im_shape), axis+(2,))

def im_2d_rot_sort_v1(im, angle=45, axis=[1]):
    """
    Return a sorted copy of an rotated image array.

    Parameters
    ----------
    im : array_like
        Input array
    angle ; 45, float
    axes : [0, 1, 2], tuple or list of ints, optional

    Returns
    -------
    sorted_im : ndarray
        Array of the same type and shape as `im`.


    """
    return ndimage.rotate(
        im_2d_sort(
                ndimage.rotate(
                    im, 
                    -angle, 
                    reshape=False,
                    mode='reflect'
                ), 
                axis=axis
            ), 
            angle, 
            reshape=False, 
            mode='reflect'
        )

def im_2d_rot_sort_v2(im, angle=45, axis=[1]):
    """
    Return a sorted copy of an rotated image array.

    Parameters
    ----------
    im : array_like
        Input array
    angle ; 45, float
    axes : [0, 1, 2], tuple or list of ints, optional

    Returns
    -------
    sorted_im : ndarray
        Array of the same type and shape as `im`.


    """
    im_shape = np.array(im.shape)

    mask_tmp = ndimage.rotate(im*0+255 , angle, reshape=True)
    mask_sort = im_2d_sort(mask_tmp, axis=axis)
    im_tmp = ndimage.rotate(im, angle, reshape=True)
    im_sort = im_2d_sort(im_tmp, axis=axis)
    
    im_tmp[mask_tmp > 0] = im_sort[mask_sort > 0]
    
    im_tmp1 = ndimage.rotate(im_tmp, -angle)
    im_tmp1_shape = np.array(im_tmp1.shape)
    im_tmp1 = np.roll(im_tmp1, -(im_tmp1_shape[:-1]- im_shape[:-1])//2, axis=(0, 1))[:im_shape[0], :im_shape[1]]


    return im_tmp1

def im_2d_rot_broken_sort_v1(im, angle=45, axis=0):
    """
    Return a sorted copy of an rotated image array.

    Parameters
    ----------
    im : array_like
        Input array
    angle : 45, float
    axes : [0, 1, 2], tuple or list of ints, optional

    Returns
    -------
    sorted_im : ndarray
        Array of the same type and shape as `im`.


    """
    im_tmp = ndimage.rotate(im, angle, reshape=False, mode='reflect')
    amask = np.argsort(im_tmp, axis=axis)
    im_sort = np.sort(im_tmp, axis=axis)

    
    return np.take_along_axis(
        im_sort, 
        amask, 
        axis=axis)