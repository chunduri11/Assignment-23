import dlib
import cv2
import numpy as np
from renderFace import renderFace
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

def writeLandmarksToFile(landmarks, landmarksFileName):
    with open(landmarksFileName, 'w') as f:
        for p in landmarks.parts():
            f.write("%s %s\n" %(int(p.x),int(p.y)))

    f.close()

	
# Landmark model location
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Get the face detector instance
faceDetector = dlib.get_frontal_face_detector()

# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# landmarks will be stored in results/family_i.txt
landmarksBasename = 'results/landmark'

number_of_frames=309

for count in range(number_of_frames):

    # Read image
    imageFilename = '5point_aligned/frame' + str(count) + '.jpg'	
    im= cv2.imread(imageFilename)


    # Detect faces in the image
    faceRects = faceDetector(im, 0)
    print("Number of faces detected: ",len(faceRects))

    # List to store landmarks of all detected faces
    landmarksAll = []

    # Loop over all detected face rectangles
    for i in range(0, len(faceRects)):
        newRect = dlib.rectangle(int(faceRects[i].left()),
                          int(faceRects[i].top()),
                          int(faceRects[i].right()),
                          int(faceRects[i].bottom()))
        # For every face rectangle, run landmarkDetector
        landmarks = landmarkDetector(im, newRect)
        # Print number of landmarks
        if i==0:
            print("Number of landmarks",len(landmarks.parts()))

        # Store landmarks for current face
        landmarksAll.append(landmarks)

        # Next, we render the outline of the face using
        # detected landmarks.
        renderFace(im, landmarks)

        # The code below saves the landmarks to 
        # results/family_0.txt … results/family_4.txt.
        landmarksFileName = landmarksBasename +"_"+ str(count)+ ".txt"
        print("Saving landmarks to", landmarksFileName)
        # Write landmarks to disk
        writeLandmarksToFile(landmarks, landmarksFileName)
	
    outputFileName = '68point_landmarks/frame' + str(count) + '.jpg'
    print("Saving output image to", outputFileName)
    cv2.imwrite(outputFileName, im)

