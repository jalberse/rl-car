import numpy as np
import cv2 as cv

def rgb_to_bw_threshold(img):
    '''
    img is a 96 x 96 x 3 matrix
    Convert the image to black and white, crop out HUD, and threshold
    Returns: 84 x 96 matrix with values of 0 (black) or 255 (white)

    Tiles within track are white, off the track is black. Region with car is also white.
    '''
    bw = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    bw = bw[0:84, 0:96] # Crop out HUD at bottom
    thresh = cv.inRange(bw,0,150)
    return thresh

    # observation is a STATE_W x STATE_H x 3 matrix
    # We convert to black and white and then threshold so that pixels on the track
    # are white (255) and grass is black (0). 
    # Where the car is also white (255).
    # We do not rescale to one-hot because it is additional computational cost with no benefit
    # as the space is still binary (0 or 255 as possible values)
    # Render to test if this encompasses track information