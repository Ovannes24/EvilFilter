import os
import numpy as np
from PIL import Image
import cv2

def gen_video_from_im_folder(image_folder):
    video_name = '.mp4'

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    image_files.sort()
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video = cv2.VideoWriter(image_folder+video_name, fourcc, 60, (width, height))

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video.write(image)

    video.release()

def gen_bubble_sort_animation(im, step=-1, mask='random', sec=10):
    dir_name = './tmp_im'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    H, W, C = im.shape
    if mask=='random':
        m = np.mgrid[0:H, 0:W][1]
        for k in range(H):
            m[k] = np.random.permutation(m[k])
    elif mask=='mean':
        m = im.mean(axis=2).astype(int)
    else:
        m = im.mean(axis=2).astype(int)


    n = len(im[0])
    for i in range(n):
        for j in range(n-i-1):
            mask = m[:, j] >= m[:, j+1]
            m[:, j][mask], m[:, j+1][mask] = m[:, j+1][mask], m[:, j][mask]
            if step%(W**2 // (sec*60))==0:
                Image.fromarray(
                    np.take_along_axis(
                        im, 
                        np.transpose(np.tile(m, (3, 1, 1)), (1,2,0)), axis=1)
                .astype(np.uint8)).save(dir_name+f"/{abs(step):016d}.jpg")
            step-=1
            if step == 0:
                break
        if step == 0:
            break
    
    gen_video_from_im_folder(dir_name)
    # os.remove(dir_name)

    return None
    