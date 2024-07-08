import argparse
import torch
import os
import numpy as np

import flower_imc as imc
import utilities as util


# Create a parser for arguments needed to predict a class

parser = argparse.ArgumentParser()

parser.add_argument('image_path', action='store', type=str, help='Path of flower image to predict')
parser.add_argument('checkpoint', action='store', type=str, default='checkpoint.pth', help='Path to checkpoint of the model for inference')
parser.add_argument('--top_k', action='store', type=int, default=3, help='Return top K most likely classes')
parser.add_argument('--category_names', action='store', type=str, default='cat_to_name.json', help='File that contains the real names of flowers')
parser.add_argument('--gpu', action='store_true', default=False, help='use GPU or MPS (Apple Silicon)')
parser.add_argument('--show_true_class', action='store_true', default=False, help='Show the true class of the image')


args = parser.parse_args()

image_path = args.image_path
checkpoint_path = args.checkpoint
top_k = args.top_k
category_names_path = args.category_names
gpu = args.gpu


# Function to predict the class of the image using our model.
def predict(image_path, checkpoint_path, top_k, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Load model from checkpoint
    model  = imc.load_checkpoint(checkpoint_path)

    # As default, use cpu that is available in any system
    device = 'cpu'

    # If use gpu is true, setup the device variable
    if gpu:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Load model on device
    model.to(device)
    

    # Enter evaluation mode
    model.eval()


    img = util.process_image(image_path) # Process the image


    # Convert image to tensor
    # If gpu is enabled use cuda.FloatTensor for image
    if gpu:
        img_tensor = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    else:
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    
    img_tensor = img_tensor.unsqueeze(dim = 0)

    # Load image to device
    img_tensor.to(device)

    
    with torch.no_grad ():
        output = model.forward(img_tensor)
    output_prob = torch.exp(output)
    
    probs, indices = output_prob.topk(top_k)
    
   
    probs = probs.cpu().numpy()
    indices = indices.cpu().numpy()

    probs = probs.tolist()[0]
    indices = indices.tolist()[0]

    class_mapping = {val: key for key, val in model.class_to_idx.items()}
    
    
    classes = [class_mapping[item] for item in indices]
    
    return probs, classes


probs, classes = predict(image_path, checkpoint_path, top_k, gpu)

# Load names of flowers classes from file
cat_to_name = util.load_classes(category_names_path)

class_names = [cat_to_name[item].title() for item in classes]

predictions = list(zip(probs, classes))
predictions = sorted(predictions, key=lambda x: x[0], reverse=True)

print(f'Class is {cat_to_name[predictions[0][1]].title()} with {predictions[0][0]*100:.2f}% probability\n')
print(f'Top {top_k} classes:')
for prediction in predictions:
    print(f'{cat_to_name[prediction[1]].title()} : {prediction[0]*100:.2f}%')

if args.show_true_class:
    print(f'True class: {cat_to_name[image_path.split("/")[-2]].title()}')

