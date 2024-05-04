import numpy as np

def mask_color_rot(im, angle=45, angle_roll=0.5):
    """
    
    """
    HEIGHT, WIDTH, COLORSIZE = im.shape
    mx, my = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))

    # th = np.pi/4

    # mx1 = mx-WIDTH//2
    # my1 = my-HEIGHT//2

    # rot_mx = ((mx1)*np.cos(th) - my1*np.sin(th) + WIDTH//2).astype(int) % WIDTH
    # rot_my = ((mx1)*np.sin(th) + my1*np.cos(th) + HEIGHT//2).astype(int) % HEIGHT
    
    im_mean = im.mean(axis=2)

    th = im_mean*(angle_roll/180)*np.pi/2 + angle*np.pi/180

    mx1 = mx-WIDTH//2
    my1 = my-HEIGHT//2

    rot_mx = ((mx1)*np.cos(th) - my1*np.sin(th) + WIDTH//2).astype(int) % WIDTH
    rot_my = ((mx1)*np.sin(th) + my1*np.cos(th) + HEIGHT//2).astype(int) % HEIGHT

    return im[rot_my, rot_mx]