import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


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

def save_checkpoint(model, save_path, arch, class_to_idx):
    
    checkpoint = {'arch': arch,
                  'classifier': model.classifier,
                  'class_idx': class_to_idx,
                  'state_dict': model.state_dict(),
                  }

    torch.save(checkpoint, save_path)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    # load variables from checkpoint
    #
    arch = checkpoint['arch']
    classifier = checkpoint['classifier']
    class_idx = checkpoint['class_idx']
    
    
    # load model    
    model = load_model(arch)
    model = models.__dict__[arch](weights = 'DEFAULT') 
    
    # Load class_idx
    model.class_to_idx = checkpoint['class_idx']
    
    for param in model.parameters():
        param.required_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model