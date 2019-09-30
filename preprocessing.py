import numpy as np
import cv2 as cv

def rgb_to_bw_threshold(img):
    '''
    img is a 96 x 96 x 3 matrix
    Convert the image to black and white, crop out HUD, and threshold (84 x 96) to 0 or 1
    Downsample to a 21x24 matrix, necessary for learning speed and memory allocation
    Then packs that into an immutable uint8 tuple for memory efficiency
    Must be immutable tuple for hashing in defaultdict

    Returns: uint8 tuple

    Pixels within track are 1, off the track is 0. Region with car is also 1.
    '''
    bw = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    bw = bw[0:84, 0:96] # Crop out HUD at bottom
    thresh = cv.inRange(bw,0,150)[::4,::4] # 84x96 0 or 255 downsample to 21x24
    cv.imshow("state",thresh)
    cv.waitKey(1)
    thresh[thresh > 0] = 1 # set to 1 if not 0
    packed = np.packbits(thresh) # Flatten and pack into uint8 array
    state = tuple(packed.tolist()) # pack into non-mutable tuple for hashing
    return state

    # observation is a STATE_W x STATE_H x 3 matrix
    # We convert to black and white and then threshold so that pixels on the track
    # are white (255) and grass is black (0). 
    # Where the car is also white (255).
    # We do not rescale to one-hot because it is additional computational cost with no benefit
    # as the space is still binary (0 or 255 as possible values)
    # Render to test if this encompasses track information
