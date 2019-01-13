import os
import numpy as np
import cv2

def expand_image_to_specific_px(filename):    
    img = cv2.imread(filename)
    print(img.shape)

    expand_img = np.zeros((img.shape[0], 512-img.shape[1], 4))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA) # change from 3 channels to 4 channels

    new_img = np.concatenate((expand_img, img), axis=1)

    cv2.imwrite(filename, new_img)


current_dir = os.getcwd()

for parent, dirnames, filenames in os.walk(current_dir):
    for filename in filenames:
        if ".png" in filename:
            expand_image_to_specific_px(filename)
