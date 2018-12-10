import numpy as np
import argparse
import imutils
import dlib
import cv2
from imutils import face_utils

shape_predictor = "/Users/bowenite/Desktop/6819_project/models/shape_predictor_68_face_landmarks.dat"

def detect(image, shape_predictor):
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)


# load the input image, resize it, and convert it to grayscale
    #image = cv2.imread(image)
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
     
# detect faces in the grayscale image
    rects = detector(gray, 1)

#stored cropped faces from image
    cropped_images = []

# loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
     
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cropped_images.append((x,y,w,h)) #image[y:y+h, x:x+w])
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
# show the output image with the face detections + facial landmarks
    #return image
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)

    return cropped_images

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    shape_predictor = "/Users/bowenite/Desktop/6819_project/models/shape_predictor_68_face_landmarks.dat"
    image = "/path/to/your/image"

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape_predictor",
            help="path to facial landmark predictor")
    ap.add_argument("-i", "--image", required=True,
            help="path to input image")
    args = ap.parse_args()

    if args.image:
        image = args.image

    detect(image, shape_predictor)
