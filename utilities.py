from PIL import Image
import numpy as np
import json

# Function to load the number of classes
def load_classes(file_path):
    with open(file_path, 'r') as f:
        class_names = json.load(f)
        return class_names
    
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
