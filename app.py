import numpy as np  
import torch
import torch.nn
import torchvision 
from torch.autograd import Variable
from torchvision import transforms
import PIL 
import cv2
from models.ResNet import *
from face_detector import *

#This is the Label
Labels = {0:"Angry",
          1: "Disgust",
          2: "Surprised",
          3: "Happy",
          4: "Sad",
          5: "Surprised",
          6: "Neutral"}

Labels = {0: "Happy",
          1: "Sad",
          2: "Surprised",
          3: "Neutral"}

# Let's preprocess the inputted frame

data_transforms = transforms.Compose(
    [
        transforms.Resize(200),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ]
) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   ##Assigning the Device which will do the calculation
model = resnet_50()
model  = model.to(device)   #set where to run the model and matrix calculation

#params = "models/7_emotions"
params = "models/5_emotions"
#params = "models/model.30"

model.load_state_dict(torch.load(params, map_location='cpu'))
model.eval()                #set the device to eval() mode for testing


#Set the Webcam 
def Webcam_720p():
    cap.set(3,640)
    cap.set(4,480)

def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = Labels[prediction]

    return result,score

def preprocess(image):
    image = PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
    #print(image)                             
    image = data_transforms(image)
    image = image.float()
    image = image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                               #accpets 4-D Vector Tensor so we need to squeeze another
    return image                            #dimension out of our 3-D vector Tensor
    
    
#Let's start the real-time classification process!
                                  
cap = cv2.VideoCapture(0) #Set the webcam
Webcam_720p()

fps = 0
show_score = 0
show_res = 'Nothing'
sequence = 0

while True:
    ret, frame = cap.read() #Capture each frame
    
    faces = detect(frame,shape_predictor)

    faces_results = []

    if faces != []:
        for face in faces:
        #faces = faces[0]
            image = frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
            #cv2.imshow("cut", image)
            image_data = preprocess(image)
            prediction = model(image_data)
            result,score = argmax(prediction)
            faces_results.append((result, score))
            print(result, score)
            print(" ")

    fps += 1
    for i, face in enumerate(faces):
        cv2.putText(frame,faces_results[i][0], (face[0],face[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
        cv2.rectangle(frame, (face[0],face[1]), (face[0] + face[2],face[1]+face[3]),(0,255,0), 2)
        

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("Face Detection")
