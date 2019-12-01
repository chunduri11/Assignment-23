import dlib
import cv2
import numpy as np
import faceBlendCommon as fbc
#from dataPath import DATA_PATH
#from dataPath import MODEL_PATH
import matplotlib.pyplot as plt
#%matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

PREDICTOR_PATH =  "shape_predictor_5_face_landmarks.dat"

# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
no_of_frames=309

for i in range(no_of_frames):

    im = cv2.imread('orig_rotated_frames/frame' + str(i) + '.jpg')

    # Detect landmarks.
    points = fbc.getLandmarks(faceDetector, landmarkDetector, im)

    points = np.array(points)

    # Convert image to floating point in the range 0 to 1
    im = np.float32(im)/255.0

    # Dimensions of output image
    h = 600
    w = 600



    # Normalize image to output coordinates.
    imNorm, points = fbc.normalizeImagesAndLandmarks((h, w), im, points)

    imNorm = np.uint8(imNorm*255)

    plt.imsave('5point_aligned/frame' + str(i) + '.jpg', imNorm[:,:,::-1])
