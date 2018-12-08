import numpy as np  
import argparse
import torch
import torch.nn
import torchvision 
from torch.autograd import Variable
from torchvision import transforms
import PIL 
import cv2
from models.ResNet import *

def run(input_file):

    data_transforms = transforms.Compose(
        [
            transforms.Resize(200),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
        ]
    )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   ##Assigning the Device which will do the calculation
    model  = resnet_50()#Load model to CPU
    model  = model.to(device)   #set where to run the model and matrix calculation
    model.load_state_dict(torch.load("models/model.9", map_location='cpu'))
    model.eval()

    image = data_transforms(input_file)
    #image = data_transforms(image)
    image = image.unsqueeze(0)

    output = model(image)
    output = output.cpu()
    output = output.detach().numpy()
    top_score = np.argmax(output, axis=1)
    score = np.amax(output)
    print(top_score, score)
    print(output)


if __name__ == '__main__':
    input_file = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to run")
    args = parser.parse_args()

    if args.image:
        input_file = args.image
    input_file = cv2.imread(input_file) 
    input_file = PIL.Image.fromarray(input_file)
    run(input_file)


