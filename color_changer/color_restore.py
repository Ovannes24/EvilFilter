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

def color_restore(im1, im2, shift=3):
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
    
    im1_mean = np.mean(im1, axis=2)
    im2_mean = np.mean(im2, axis=2)

    
    im2_list = []
    h, w = im1_mean.shape

    for i in range(0, h):
        a = im1_mean[i]
        b = im2_mean[i]

        im2_list.append(restore_line_pixels(a, b, shift))

    im2_list = np.array(im2_list)
    # im1 = im2_list


    return np.take_along_axis(
            im1, 
            np.transpose(np.tile(im2_list, (3, 1, 1)), (1,2,0)), 
            axis=1
        )