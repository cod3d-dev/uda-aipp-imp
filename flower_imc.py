import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

# Function to load the number of classes
def load_classes(file_path):
    with open(file_path, 'r') as f:
        class_names = json.load(f)
        num_classes = len(class_names)
        return class_names, num_classes
    
# Function to load the model
def load_model(arch, output_units = 102, hidden_units = 1000):
# Load the model depending on the architecture
    model = models.__dict__[arch](weights = 'DEFAULT') 


    if arch[:3] == 'res':
        num_features = model.fc.in_features
    elif arch == 'alexnet':
        num_features = model.classifier[1].in_features
    elif arch[:5] == 'dense':
        num_features = model.classifier.in_features
    else:
        num_features = model.classifier[0].in_features

    # Freeze pretrained model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define our classifier
    classifier = nn.Sequential(nn.Dropout(0.4),
                            nn.Linear(num_features, hidden_units),
                            nn.ReLU(),
                            nn.Linear(hidden_units, output_units),
                            nn.LogSoftmax(dim=1))

    # Replace the pretrained model classifier with our classifier depending if the pretrained uses fc or classifier
    if arch[:3] == 'res':
        model.fc = classifier
    else:
        model.classifier = classifier

    return model

def save_checkpoint(model, save_path, optimizer_state_dict, num_epochs, arch, hidden_units, output_units, class_to_idx):
    
    checkpoint = {'arch': arch,
                  'classifier': model.classifier,
                  'optimizer_state_dict': optimizer_state_dict,
                  'class_idx': class_to_idx,
                  'state_dict': model.state_dict(),
                  'num_epochs': num_epochs,
                  'hidden_units': hidden_units,
                  'output_units': output_units
                }

    torch.save(checkpoint, save_path)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    # load variables from checkpoint
    #
    arch = checkpoint['arch']
    num_epochs = checkpoint['num_epochs']
    hidden_units = checkpoint['hidden_units']

    # load model    
    model = load_model(arch, hidden_units = hidden_units)
    model = models.__dict__[arch](weights = 'DEFAULT') 
    
    # Load class_idx
    model.class_to_idx = checkpoint['class_idx']
    
    for param in model.parameters():
        param.required_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])



    # Load optimizer state dictionary
    optimizer_state_dict  = checkpoint['optimizer_state_dict']

    return model, num_epochs, optimizer_state_dict

# Function to process image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Load image
    img = Image.open(image_path)

    # Process a PIL image for use in a PyTorch model

    # Resize image based on shortes side
    size = img.size

    if size[0] <= size[1]:
        cpercent = 256 / size[0]
        new_size = [256, int(size[1]*cpercent)]
    else:
        cpercent = 256 / size[1]
        new_size = [int(size[0]*cpercent), 256]

    new_img = img.resize((new_size[0], new_size[1]), Image.Resampling.LANCZOS)
    
    # center crop image
    left = (new_img.size[0] - 256)/2
    top = (new_img.size[1] - 256)/2
    right = (new_img.size[0] + 256)/2
    bottom = (new_img.size[1] + 256)/2
    
    new_img = img.crop((left, top, right, bottom))
    # normalize
    img_processed = np.array(img)/255

    means = [0.485, 0.456, 0.406]
    stdev = [0.229, 0.224, 0.225]
    
    img_processed = (img_processed - means) / stdev
    img_processed = img_processed.transpose(2, 0, 1)

    return img_processed
