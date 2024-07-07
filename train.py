import argparse
import matplotlib.pyplot as plt
import math
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import os
from PIL import Image
import numpy as np
import json

import flower_imc as imc


# Create a parser that accepts --save_dir and --archand --learning_rate and --hiden_units and --epochs and --gpuj

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', action='store', type=str, help='Path of directory to train')
parser.add_argument('--save_dir', action='store', type=str, default='checkpoint.pth', help='File path to save a checkpoint of the model')
parser.add_argument('--arch', action='store', type=str, default='vgg16_bn', help='what model to use: alexnet | densenet161 | densenet169 | densenet201 | vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | resnet50 | resnet101 | resnet152')
parser.add_argument('--learning_rate', action='store', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_units', action='store', type=int, default=1000, help='number of hidden units')
parser.add_argument('--epochs', action='store', type=int, default=1, help='number of epochs to train')
parser.add_argument('--gpu', action='store_true', default=False, help='use GPU or MPS (Apple Silicon)')
parser.add_argument('--output_units', action='store', default=102, help='Number of classes in the training dataset')                    
parser.add_argument('--batch_size', action='store', default=80, help='Define the batch size')
parser.add_argument('--resume', action='store_true', default=False, help='Resume training')


args = parser.parse_args()

# Define variables for arguments passed by the user
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu
batch_size = args.batch_size
resume = args.resume
output_units = args.output_units

# Transformations for training, validation, and test
train_transforms = transforms.Compose([transforms.RandomRotation(60),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_dataset = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
validation_dataset = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)

# Define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4)
validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, num_workers=4)



# Define tracking variables
start_time = time.time()
steps = 0
running_loss = 0
print_every = 40
start_epoch = 0

# Load model and checkpoint to resume training
if resume:
    model, start_epoch, optimizer_state_dict,  = imc.load_checkpoint(save_dir)
else: 
    # Load the model
    model = imc.load_model(args.arch, output_units, hidden_units)

    # Setup the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# Load model to GPU or MPS (Apple Silicon)
# As default, use cpu that is available in any system
device = 'cpu'

if gpu:
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    num_workers = 10
else:
    num_workers = 0
    
# Move model to GPU or MPS (Apple Silicon)
model.to(device)
if resume:
    optimizer.load_state_dict(optimizer_state_dict)

print('Flower Classifier\n',
      f'Architecture: {arch}\n',
      f'Learning Rate: {learning_rate}\n',
      f'Hidden Units: {hidden_units}\n',
      f'Total Epochs: {epochs}\n',
      f'Starting Epoch: {start_epoch}\n',
      f'Device: {device}\n')


for epoch in range(start_epoch, epochs):
    steps = 0
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # Reset gradients
        
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval() # Setup model in evaluation mode
            valid_loss = 0
            accuracy = 0
            
            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    
                    # Calculate accuracy with validation dataset
                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"TR loss: {running_loss/print_every:.3f} | "
                  f"V loss: {valid_loss/len(validationloader):.3f} | "
                  f"V Acc: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            model.train()
print(f"\nEnd Epoch {epoch+1} | Total Time: {(time.time() - start_time):.2f} \n ")

# Save checkpoint
# Get the mapping of classes to indexes to store in checkpoint for inference
class_to_idx = train_dataset.class_to_idx
imc.save_checkpoint(model, save_dir, optimizer.state_dict(), epochs, arch, hidden_units, output_units, class_to_idx)


# Do validation on the test set

# Variables to track accuracy
accuracy = 0
test_loss = 0
 # Setup model in evaluation mode

with torch.no_grad():
    model.eval()
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
     
        outputs = model.forward(inputs)
        test_loss += criterion(outputs, labels)
        
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
        
print(f'Test accuracy: {100 * accuracy/len(testloader):.2f}%')
