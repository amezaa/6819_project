import cv2
import os.path
from faced import FaceDetector
from faced.utils import annotate_image

# #This code is used for splitting a video into individual frames
# vidcap = cv2.VideoCapture('bill.avi')
# success,image = vidcap.read()
# count = 0
# while success:
#     vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 2000))  # save frame every (count*200) milli seconds
#     cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#     success,image = vidcap.read()
#     print('Read a new frame: ', success)
#     count += 1

face_detector = FaceDetector()
count = 1
group_img = "098.jpg" # "frame%d.jpg" % count
img = cv2.imread(group_img) #frame%d.jpg" % count

rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
# Receives RGB numpy image (HxWxC) and
# returns (x_center, y_center, width, height, prob) tuples.
bboxes = face_detector.predict(rgb_img, 0.8) #If lover than 0.8 the network will also find small face portions in the background
    #ann_img = annotate_image(ann_img,[(30, 30, 30, 30, 0)])

num_face = 0
boxPer = 1.1
while num_face <len(bboxes):
    bleft = (bboxes[num_face][0]-round(boxPer*bboxes[num_face][2]/2))
    bright = ((bboxes[num_face][0]-round(boxPer*bboxes[num_face][2]/2)+bboxes[num_face][2]))
    btop = (bboxes[num_face][1]-round(boxPer*bboxes[num_face][3]/2))
    bbottom = ((bboxes[num_face][1]-round(boxPer*bboxes[num_face][3]/2)+bboxes[num_face][3])+10)
    img_face = img[btop:bbottom,bleft:bright]

    cv2.imwrite("boxed_face%d.jpg" % num_face,img_face)
    num_face += 1
    print('Boxed a new frame: ', num_face)

ann_img = annotate_image(img, bboxes) # Use this utils function to annotate the image.
cv2.imwrite("boxed_frame.jpg",ann_img)



