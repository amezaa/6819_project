import cv2
import os.path
from faced import FaceDetector
from faced.utils import annotate_image

vidcap = cv2.VideoCapture('bill.avi')
success,image = vidcap.read()
count = 0
while success:
    vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 2000))  # save frame every (count*200) milli seconds
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

face_detector = FaceDetector()
count2 = 0
print(count2)
while count2 < count:
    print("frame%d.jpg" % count2)
    img = cv2.imread("frame%d.jpg" % count2) #frame%d.jpg" % count
    rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    # Receives RGB numpy image (HxWxC) and
    # returns (x_center, y_center, width, height, prob) tuples.
    bboxes = face_detector.predict(rgb_img, 0.3)
    print(bboxes)
    ann_img = annotate_image(img, bboxes) # Use this utils function to annotate the image.
    #ann_img = annotate_image(ann_img,[(30, 30, 30, 30, 0)])
    print(bboxes[0][1]-round(bboxes[0][3]/2))
    print(bboxes[0][1]-round(bboxes[0][3]/2)+bboxes[0][3])
    img_face = img[(bboxes[0][1]-round(bboxes[0][3]/2)):((bboxes[0][1]-round(bboxes[0][3]/2)+bboxes[0][3])),(bboxes[0][0]-round(bboxes[0][2]/2)):((bboxes[0][0]-round(bboxes[0][2]/2)+bboxes[0][2]))]
    cv2.imshow("cropped", img_face)
    cv2.waitKey(0)
    cv2.imwrite("boxed_frame%d.jpg" % count2,ann_img)
    print('Boxed a new frame: ', count2)
    count2 += 1




#     img = cv2.imread("frame%d.jpg" % count2) #frame%d.jpg" % count
#     rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
#     # Receives RGB numpy image (HxWxC) and
#     # returns (x_center, y_center, width, height, prob) tuples.
#     bboxes = face_detector.predict(rgb_img, 0.5)
#             #print(bboxes)
#     ann_img = annotate_image(img, bboxes) # Use this utils function to annotate the image.
#     cv2.imwrite("boxed_frame%d.jpg" % count2,ann_img)
#     print('Boxed a new frame: ', count2)
#     count2 += 1

