import numpy as np

def SE(a, b):
    return (a-b)**2

def AE(a, b):
    return np.abs(a-b)

def restore_line_pixels(a, b, shift=3):
    al = len(a)
    index = np.arange(0, al)
    
    shift_index = np.argmin(
        np.array([AE(a, np.roll(b, i)) for i in range(-(shift//2), (shift//2)+1)]),
        axis=0
    ) - (shift//2)

    argmask = (index + shift_index) % al

    return argmask

def color_restore(im, shift=3):
    """
    Return a color restore of im by shift.

    Parameters
    ----------
    im1 : array_like
        Input array
    shift : int


    Returns
    -------
    im : ndarray
        Array of the same type and shape as `im` and `im_mask`.


    """
    
    im1 = np.mean(im, axis=2)

    
    im1_list = []
    h, w = im1.shape

    for i in range(0, h):
        a = im1[i-1]
        b = im1[i]

        im1_list.append(restore_line_pixels(a, b, shift))

    im1_list = np.array(im1_list)
    im1 = im1_list


    return np.take_along_axis(
            im, 
            np.transpose(np.tile(im1, (3, 1, 1)), (1,2,0)), 
            axis=1
        )