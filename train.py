import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *

def run():
    # Parameters
    num_epochs = 20
    output_period = 100	
    batch_size = 15

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_50()
    model = model.to(device)
    model.load_state_dict(torch.load("models/model.10"))

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    #optimizer = optim.SGD(model.parameters(), lr=.1)
    optimizer = optim.SGD(model.parameters(), lr=.003, momentum=0.9)

    epoch = 10
    while epoch <= num_epochs:
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                epoch, batch_num*1.0/num_train_batches,
                running_loss/output_period
                ))
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/model.%d" % epoch)

        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here
        if epoch%1 == 0 :
            model.eval()
       #top-1 error
            total = 0
            correct = 0
            correct_5 = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
                _, predicted5 = torch.topk(outputs.data, 5)
                for i in range(len(labels)):
                    if labels[i] in predicted5[i]:
                        correct_5 += 1
	
            accuracy_t = (float(correct)/float(total))*100
            accuracy_t_5 = (float(correct_5)/float(total))*100
            print("Top 1 Accuracy on training data for epoch #" + str(epoch) + " = " + str(accuracy_t) + "%")
            print("Top 5 Accuracy on training data for epoch#" + str(epoch) + " = " + str(accuracy_t_5) + "%")
        
            total = 0
            correct = 0
            correct_5 = 0
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

             
                _, predicted5 = torch.topk(outputs.data, 5)
                for i in range(len(labels)):
                    if labels[i] in predicted5[i]:
                        correct_5 += 1

            accuracy_v = (float(correct)/float(total))*100
            accuracy_v_5 = (float(correct_5)/float(total))*100
            print("Top 1 Accuracy on validation data for epoch #" +str(epoch) + " = " + str(accuracy_v) + "%")
            print("Top 5 Accuracy on validation data for epoch #" + str(epoch) + " = " + str(accuracy_v_5) + "%")

            gc.collect()
        epoch += 1

print('Starting training')
run()
print('Training terminated')
