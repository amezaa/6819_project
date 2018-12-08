import cv2
import os.path
from faced import FaceDetector
from faced.utils import annotate_image

face_detect = FaceDetector()
count = 1
print(count)

while os.path.isfile("convert/Confused_%d.jpg" % count):
    print(os.path.isfile("convert/Confused_%d.jpg" % count))
    print("convert/Confused_%d.jpg" % count)
    img = cv2.imread("convert/Confused_%d.jpg" % count) #frame%d.jpg" % count
    rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    # Receives RGB numpy image (HxWxC) and
    # returns (x_center, y_center, width, height, prob) tuples.
    bboxes = face_detect.predict(rgb_img, 0.7)
    print(bboxes)
    #ann_img = annotate_image(ann_img,[(30, 30, 30, 30, 0)])
    #print(bboxes[0][1]-round(bboxes[0][3]/2))
    #print(bboxes[0][1]-round(bboxes[0][3]/2)+bboxes[0][3])
    if len(bboxes)>0:
        img_face = img[(bboxes[0][1]-round(bboxes[0][3]/2)-10):((bboxes[0][1]-round(bboxes[0][3]/2)+bboxes[0][3])+10),(bboxes[0][0]-round(bboxes[0][2]/2)-10):((bboxes[0][0]-round(bboxes[0][2]/2)+bboxes[0][2])+10)]
        cv2.imwrite("Conf/Conf_face_%d.jpg" % count, img_face)

    ann_img = annotate_image(img, bboxes) # Use this utils function to annotate the image.
    cv2.imwrite("Conf/Conf_%d.jpg" % count,ann_img)

    count += 1


