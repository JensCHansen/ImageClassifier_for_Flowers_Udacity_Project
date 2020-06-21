#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */utility_functions.py
       
##################################################################################
# PROGRAMMER: Jens Hansen
# DATE CREATED: 06.06.2020                            
# REVISED DATE: 
# PURPOSE: Consists of several functions necessary to train the dataset:
#          1) Get input arguments
#          2) Get data

##################################################################################

## Import necessary python modules
import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F

from PIL import Image
import numpy as np
from os import listdir

import matplotlib.pyplot as plt

from collections import OrderedDict

import ast

import json

import os
import random


## Functions defined below:

## Utility functions

def get_input_args():
    """
    Retrieves and parses different command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. 
    
    Mandatory:
      0. Path to folder of images as data_directory
      
    Optional:
      1. Directory for saving checkpoint as --save_dir with default "ImageClassifier/"
      2. Architecture of the network as --arch with default "vgg13"
      Hyperparameters learning rate, # of hidden layers and # of epochs as
      3. --learning_rate with default "0.01"
      4. --hidden_units with default "512"
      5. --epochs with default "20"
      6. GPU usage --gpu with default "on"
    
    This function returns these arguments as an ArgumentParser object.
    
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    
    ## Create parse
    msg = ('Define the data directory, the saving directory, the model architecture,' 
           ' and the model hyperparameters for your neural network.')
        
    parser = argparse.ArgumentParser(description=msg)
    
    ## Creates command line arguments:
    
    # Directory with images
    parser.add_argument('data_directory', type=str, default='ImageClassifier/', 
                        help='path to folder of images')
    
    # Directory for saving checkpoint
    parser.add_argument('-s', '--save_dir', type=str, default='ImageClassifier/checkpoint.pth', 
                        help='"path/to/filename.pth" to save trained network')
    
    # Batch size
    parser.add_argument('-b', '--batch_size', type=int, default = 64,
                       help='image batch size for dataloader')
    
    # Model architecture
    parser.add_argument('-a', '--arch', type=str, default = 'vgg13',
                       help='model architecture of neural network')
    
    # Hyperparameters
    describe = 'hyperparameter of neural network'
    parser.add_argument('-lr', '--learning_rate', type=float, default = 0.001, help=describe)
    parser.add_argument('-hu', '--hidden_units',  type=int,   default = 512,   help=describe)
    parser.add_argument('-ep', '--epochs',        type=int,   default = 4,     help=describe)
                        
    # GPU mode
    parser.add_argument('-g', '--gpu', type=str, choices=['on','off'], default = 'on', help='GPU mode off/on')
    
    ## Store inputs to variable
    in_args = parser.parse_args()
    
    ## Check inputs
    
    # Check 1: input
    save_dir = in_args.save_dir
    
    # Check for file extension '.pth'
    if save_dir[-4:] != '.pth':
        
        # Remove any '.' from input
        save_dir = save_dir.split('.')[0] # Get part of string before '.'
        
        # Add proper file extension to string and overwrite 'in_args.save_dir'
        in_args.save_dir = save_dir + '.pth'
    
    ## Return parsed argument collection that was created with this function 
    return in_args


def create_model_list():
    
    """
    Function creates a list of names of pre-trained models from 
    torchvision.models. Only the models are included which have 
    pre-trained versions. Trained version of "Inception" and "Squeezenet" 
    not included because of different / more complicated input / output 
    treatments (-> tu be added in the future ;-).
    
    Parameters:
     None
    Returns:
     model_list - list with pre-trained models from torchvision.models.
    """
        
    # Set start index (only include relevant attributes)
    start_idx = 0 
    
    # Return a list of valid attributes of the object 'models'
    model_list = dir(models)
    
    # Only use relevant list entries (here: start to end - all included)
    model_list = model_list[start_idx:]
    
    # Specifiy a list with terms that should be dropped from model list
    drop_list = ['densenet', 'inception', 'resnet', 'squeezenet', 'vgg',
                 'AlexNet', 'DenseNet', 'Inception3', 'ResNet', 'SqueezeNet', 'VGG',
                 '__builtins__', '__cached__', '__doc__', '__file__', '__loader__',
                 '__name__', '__package__', '__path__', '__spec__', 'inception_v3', 
                 'squeezenet1_1', 'squeezenet1_0' ]
    
    # Loop over the drop_list and remove items from the model list
    for drop in drop_list:
        if drop in model_list:
            model_list.remove(drop)
    
    # Return the final model list
    return model_list


def return_resnet_models():
    """
    Simpy returns the different pre-trained resnet model names.
    """
    
    resnet = ['resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50']
    
    return resnet
 
    
def load_data(data_directory, bts, input_size):
    
    """
    Loads image data from sprecified directory (the dataset needs to be split into 
    three parts, training, validation (respetive folders 'train', 'valid' and 'test' are 
    necesseray). 
    
    For the training, transformations such as random scaling, cropping, and flipping are
    applied. Input data is resized to 224x224 pixels as required by the pre-trained 
    networks.
    
    The validation and testing sets are resized and cropped to appropoate sizes.
    For all three sets the means and standard deviations of the images are normalized
    to what the network expects. For the means, it's [0.485, 0.456, 0.406] and for the 
    standard deviations [0.229, 0.224, 0.225], calculated from the ImageNet images. 
    These values will shift each color channel to be centered at 0 and range from -1 to 1.
    Dataloaders for training, validation and testing data are defined.
    The function returns the transformed image data sets and the dataloaders.
    
    Parameters:
     data_directory - main directory where the image data can be found
     bts            - batch size: number of images per batch
     input_size     - required image input size by the model architecture
    Returns:
     dataloader     - provides transformed image data for training, validation and testing
     datasets       - transformed image data for training, validation and testing
    """
    
    # Quick check if '/' was added at the end of directoy path
    if data_directory[-1] == '/':
        data_directory = data_directory[:-1]
    
    # Define paths to train, validation and test data
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir  = data_directory + '/test'
    
    # Define transformations for the data sets
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(input_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(input_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])

    val_transform = test_transform

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transform)

    # Create a dictionay with the loaded data
    image_datasets = {'train': train_dataset,
                      'valid': valid_dataset,
                      'test':  test_dataset}

    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bts, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bts, shuffle=True)
    testloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=bts, shuffle=True)
    
    # Create a dictionay with the loaded data
    dataloaders = {'train': trainloader,
                   'valid': validloader,
                   'test':  testloader}
    
    return dataloaders, image_datasets


def label_mapping(category_names = 'cat_to_name.json'):
    
    """
    Function loads in a mapping from category label to category name and creates
    a dictionary from it where the keys are the indices and the values the names.
    
    Parameters:
     category_names - name of the file (has to be in the folger "ImageClassifier"
    Returns:
     cat_to_name - dictionary mapping the integer encoded categories to the actual 
     names of the flowers.
    """
    
    ## Create string
    path_category_names = 'ImageClassifier/' + category_names
    
    ## Load in a mapping from category label to category name (JSON object)
    with open(path_category_names, 'r') as f:
        name_mapping = json.load(f)
        
        # Create a dictionary where the keys are the numbers (converted from string
        # to integers) and the values are the names
        name_mapping = {int(k):v for k,v in name_mapping.items()}
    
    ## Return the dictionary
    return name_mapping


def set_parameter_requires_grad(model, feature_extracting):
    """
    Function from: pytorch.org - Finetuning torchvision models tutorial
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    
    This helper function sets the ".requires_grad" attribute of the parameters in the model
    to "False" when we are feature extracting. By default, when we load a pretrained 
    model all of the parameters have ".requires_grad=True", which is fine if we are training 
    from scratch or finetuning. However, if we are feature extracting and only want to
    compute gradients for the newly initialized layer then we want all of the other 
    parameters to not require gradients
    
    Parameters:
     model - the chosen model architecture
     feature_extracting - Boolean variable, when "True" no gradients for all existing 
                          layers of the model
    Returns:
     None
    """
    
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def check_gpu(gpu):
    
    ## Set GPU mode 
    if gpu == 'on':
        
        # Use GPU if it's available
        device = "cuda"
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    else:
        device = "cpu"
        
    return device


def get_input_predict():
    """
    Retrieves and parses different command line arguments provided by the user when
    they run the program 'predict.py' from a terminal window. This function uses Python's 
    argparse module to create and define these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. 
    
    Basic usage:
    python predict.py /path/to/image checkpoint

    1) path_to_image                  - specifiy path to the image
    2) checkpoint                     - checkpoint file with trained model 
                                        saved from train.py

    Options:
    python predict.py /path/to/image checkpoint --option, see below

    1) --top_k N                      - returns the most likely classes N, 
                                        with N being an integer
    2) --category_names filename.json - use a mapping of categories to real
                                        names
    3) --gpu                          - use GPU inteference
    
    This function returns these arguments as an ArgumentParser object.
    
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """
    
    ## Create parse
    msg = ('Load model from checkpoint and predict classes for image.')
    
    parser = argparse.ArgumentParser(description=msg)
    
    ## Creates command line arguments:

    # Mandatory input 
    
    # Path to image
    default_image_path = 'Random'
    
    parser.add_argument('path_to_image', type=str, default=default_image_path, 
                        help='Path to images, if set to "Random" a path to a '
                             'random image from the test data will be set (=default).')
    
    # Path to model checkpint
    default_checkpoint = 'ImageClassifier/checkpoint.pth'
    
    parser.add_argument('checkpoint', type=str, default=default_checkpoint, 
                        help='Model checkpoint to be loaded')
    
    # Optional
    
    # Directory for saving checkpoint
    parser.add_argument('--top_k', type=int, default= 3, 
                        help='Number of most likely classes to be displayed')
    
    # Mapping of categories to real name
    parser.add_argument('-cn', '--category_names',  type=str, default = 'cat_to_name.json',
                       help='File name with mapping of categories to real name'
                            '(please note: has to be in folder "ImageClassifier/")')
    
    # GPU mode
    parser.add_argument('-g', '--gpu', type=str, choices=['on','off'], default = 'on', help='GPU mode off/on')
    
    # Assigns variable in_args to parse_args()
    in_args_predict = parser.parse_args()
    
    ## Feature: if 'Random' was typed in assign path of random image from test data (= default)
    if in_args_predict.path_to_image == 'Random':
        
        # Assign random image path from test images to variable
        in_args_predict.path_to_image = random_test_load()
    
    # Return parsed argument collection that was created with this function 
    return in_args_predict


def save_model_checkpoint(image_datasets, model, optimizer, criterion, in_arg):
    
    """
    Function which saves the model so it can be loaded later for making predictions.
     
    Parameters:
     arch - 
    Returns:
     model - 
     
    """
    
    # The parameters for PyTorch networks are stored in a model's state_dict
    #print("Our model: \n\n", model, '\n')
    #print("The state dict keys: \n\n", model.state_dict().keys())
    #print(optimizer.state_dict().keys())
    
    # Directory for saving checkpoint
    save_dir = in_arg.save_dir
    
    # Save model architecture (name for reload it):
    model.arch = in_arg.arch

    # Mapping info
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    resnet = return_resnet_models()
    
    # Set checkpoint diczionary
    checkpoint = {'model_arch': model.arch,
                  'input_size': model.input_size,
                  'model_state_dict': model.state_dict(),
                  'mapping': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'criterion': criterion}
    
    # Distinguish btw. model which 'classifer' and fc 'layer':
    if model.arch in resnet:
        checkpoint.update( {'classifier' : model.fc} )
        
    else:
        checkpoint.update( {'classifier' : model.classifier} )


    torch.save(checkpoint, save_dir)
    
    
def load_model_checkpoint(model_path):

    ''' 
    Load a pre-trained deep learning model which was saved with function
    "save_model_checkpoint()".
    
    Parameters:
     model_path   - full path to the saved model including the filename with the saved 
                    model data
    
    Returns:
     model        - deep learning model saved under "model_path" with the function 
                    "save_model_checkpoint()"
     criterion    - loss function with which the loaded model was trained (also saved along
                    with the model with the function "save_model_checkpoint()"
    
    '''
    
    # Load checkpoint data, ,add all tensors onto the CPU, using a function
    # Source: https://pytorch.org/docs/master/generated/torch.load.html      
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    ## Rebuild the model
    
    # Get model architecture
    arch = checkpoint['model_arch']
    
    # Load pretrained model with spedified model architecture
    model = getattr(models, arch)(pretrained=True)
    
    # Save architecture name to model:
    model.arch = arch
    
    # Load image input size required by specified model architecture
    model.input_size = checkpoint['input_size']
    
    # Load classifier of the model
    resnet = return_resnet_models()
    
    if arch in resnet:
        
        model.fc = checkpoint['classifier']
    
    else:
        model.classifier = checkpoint['classifier']
    

    # Load the state dict in to the network
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load mapping
    model.class_to_idx = checkpoint['mapping']
    
    # Load criterion
    criterion =  checkpoint['criterion']

    # Return model and criterion
    return model, criterion


def process_images(image, input_size):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array.
    
    Parameters:
     image        - PIL image 
     input_size   - input size for the image required by the model
    
    Returns:
     np_image     - Numpy array of image 
    
    '''
    
    # Define image transofrmations (resize, crop, tensor conversion, normalize)
    transform_image = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(input_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    # Apply transformations to image
    np_image = transform_image(image)
    
    # Return Numpy array of image
    return np_image

    
def compile_results(top_p, top_class, name_mapping):
    
    ''' 
    Function compiles the probabilities from an image inference model with the 
    corresponding image classes and returns a dictionary with the image classes as 
    keys and the probabilities as values.
        
    Parameters:
     probabilities - tuple consisting of two torch tensors (from function predict()), 
                     first  tensor: index for class mapping, 
                     second tensor: probability calculated by model
     name_mapping  - dictionary with idices as keys and corresponding class names
                     as values.

    Returns:
     results_dic   - dictionary with the image classes as keys and the 
                     probabilities as values
    '''
    
    ## Get probabilities and convert to list
    p = top_p[0].cpu().numpy().tolist()
    
    ## Create list with indices which can then be mapped to the corresponding names
    keys = top_class[0].cpu().numpy().tolist()
    
    ## Create a list with class names:
    
    # Interate thru dictionary 'name_mapping using the keys to get corresponding 
    # class names
    class_names = get_class_name(keys, name_mapping)
 
    ## Create results dictionary:
    
    # Initialize empty dictionary for the resultes   
    results_dic = dict()
    
    # Create dictionary entries
    for i in range(len(class_names)):
        if class_names[i] not in results_dic:
            results_dic[class_names[i]] =  float(p[i]) # convert probabilit 
    
    print('Indices: ', keys)
    #print(p)       
    # print(class_names)
    print('result dictionary :', results_dic)

    ## Return the results dictionary
    return results_dic, keys


def random_test_load(directory = './ImageClassifier/flowers/test/'):
    
    ''' 
    Loads a random image from a specified directory and returns its image path.
      
    Parameters:
    directory        - path to top directory with image files

    Returns:
     random_filepath - filepath to random image
    '''   

    ## Initialize empty list
    filelist = []
    
    ## Loop over subdirectories of folder with test images
    for root, dirs, files in os.walk(directory):
        
        # Loop over files:
        for file in files:    
            
            # Check for file ending '.jpg'
            if file.endswith('.jpg'):
                
                # Construct full path of image
                filepath = root + '/' + file
                
                # Append filepath of image to list
                filelist.append(filepath)
    
    ## Chose a random image path
    random_filepath = random.choice(filelist)
    
    ## Return filepath to random image
    return random_filepath
    
    
def get_index_from_image_path(image_path):
    
    # Split sting at '/' to a list of items and get 
    # the penultimate entry from that list
    img_idx = image_path.split('/')[-2]

    # Return the label
    return int(img_idx)
    
    
def get_class_name(img_indices, name_mapping):
    
    if isinstance(img_indices, list):
        
        class_names = [name_mapping.get(idx) for idx in img_indices]
        
    elif isinstance(img_indices, int):
        class_names = name_mapping[img_indices]
        
    else:
        print("Input must be either integer with single index or list of indices")
    
    return class_names

    
# Function directly related to the deep learning model

def initialize_model(arch, hidden_units, feature_extract=True):
    
    """
    Function for initializing the chosen model architecture. For this the following
    operations are conducted:
    1) Load the pre-trained model
    2) Sets gradient on/off for parameters of the existing model layers (default: no
       gradients required for existing model layers (only feature extraction - train
       the newly added layers)
    3) Specifiy input size of the images in variabel "input_size" (specific for model
       architecture)
    4) Create a new classifier and replace either the existing one of the model or
       replace the final layers of the existing model (specific for model
       architecture)
    5) Return updated model archictecture and required input size for the model.
    
    Parameters:
     arch - name of the choosen model architecture
     feature_extract - Boolean variable, default: "True", i.e. no gradients for all 
                       existing layers of the model
    Returns:
     model - updated model architecture
     input_size - required image size for input to the model
    """
    
    # Load pre-trained model from torchvision.models with attribute saved in 'arch'
    model = getattr(models, arch)(pretrained=True)
    
    # Set gradients off for existing model layers (only feature extraction)
    set_parameter_requires_grad(model, feature_extract)
    
    # Print the model architecture
    print('\nThis is the architecture of {}: \n{}\n '.format(arch, model))
    
    # Specify input size:
    input_size = 224
    
    ## Models w/o classifier:
    resnet = return_resnet_models()
    
    ## If architecture 'Resnet' was chosen
    if arch in resnet:
        
        # Get number of inputs for "fc" layer:
        in_classifier = model.fc.in_features
        out_fc1 = int(in_classifier/2)
        
        # Delete attribute 'fc' from model
        #model.fc = Identity()
        #delattr(model, 'fc')
        
        # Define the new classifier with a OrderedDict 
        model.fc = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(in_classifier, out_fc1)),
                                  ('relu1', nn.ReLU()),
                                  ('drop1', nn.Dropout(p=0.3)),
                                  ('fc2', nn.Linear(out_fc1, hidden_units)),
                                  ('relu2', nn.ReLU()),
                                  ('drop2', nn.Dropout(p=0.3)),
                                  ('fc3', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

        
    ## Models with classifier:
    else: 
        
        # Specify input size:
        input_size = 224
        
        # Create list with layers of the classifier 
        # Code inspired from: 
        # https://discuss.pytorch.org/t/how-to-get-layer-index-by-name-in-nn-sequential/25188/2
        list_classifier =  list( dict( model.classifier.named_children() ).values() )
        
        # Check number of layers in classifier
        num_classifier = len(list_classifier)
        
        # If classifier consists only of one layer, e.g. model densenet
        if num_classifier == 0: 
            
            # Get number of inputs for existing classifier the model:
            in_classifier = model.classifier.in_features
            out_fc1 = int(in_classifier/2)
            
        
        # If classifier consists of more than one layer, e.g. we have to find first layer of classifier
        else: 
            
            # Get list of indices from the classifier list of type 'Linear' Or 'Conv2d'
            # Code inspired from:
            # https://stackoverflow.com/questions/152580/whats-the-canonical-way-to-check-for-type-in-python
            # https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    
            indices = [i for i, s in enumerate(list_classifier) if isinstance(s, nn.modules.linear.Linear) 
                                                                or isinstance(s, nn.modules.conv.Conv2d)]
        
            # Get index of first linear layer of existing classifier of the model
            idx_lin = indices[0]
    
            # Get number of inputs for existing classifier the model:
            in_classifier = model.classifier[idx_lin].in_features
            out_fc1 = int(in_classifier/2)
            

        # Define the new classifier with a OrderedDict      
        model.classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(in_classifier, out_fc1)),
                                  ('relu1', nn.ReLU()),
                                  ('drop1', nn.Dropout(p=0.3)),
                                  ('fc2', nn.Linear(out_fc1, hidden_units)),
                                  ('relu2', nn.ReLU()),
                                  ('drop2', nn.Dropout(p=0.3)),
                                  ('fc3', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

    model.input_size = input_size
    
    print('This is the new model:\n{}.'.format(model))

        
    return model


def train_model(model, dataloaders, image_datasets, in_arg):
    
    ## Get data
    trainloader = dataloaders.get('train')
    validloader = dataloaders.get('valid')
    
    ## Check if GPU mode is available 
    device = check_gpu(in_arg.gpu)
        
    ## Define loss function
    criterion = nn.NLLLoss()

    ## Define optimizer
    # Only train the classifier parameters, feature parameters are frozen
    
    ## Models w/o classifier:
    resnet = return_resnet_models()
    
    # For resnet, the classifier is in 'fc' layer
    if in_arg.arch in resnet:
        optimizer = optim.Adam(model.fc.parameters(), in_arg.learning_rate)
    
    # For resnet, the classifier is in 'classifier' layer
    else:
        optimizer = optim.Adam(model.classifier.parameters(), in_arg.learning_rate)

    model.to(device)
    
    epochs = int(in_arg.epochs)
    steps = 0
    running_loss = 0
    print_every = 30

    for epoch in range(epochs):
    
        for inputs, labels in trainloader:
            steps += 1
        
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Set the parameter gradients to zero
            optimizer.zero_grad()
            
            # Set model to training mode
            model.train()  
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            # Feed backwards and optimize
            loss.backward()
            optimizer.step()
            
            # Add to running loss
            running_loss += loss.item()
        
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                
                # Set model to evaluate mode
                model.eval()
            
                # Turn off gradients for faster performance
                with torch.no_grad():
                
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        val_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Training loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {val_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader)*100:.3f} %")
            
                running_loss = 0
                model.train()
    
    return model, optimizer, criterion


def test_model(dataloaders, model, gpu, criterion):
    
    # Get data
    testloader = dataloaders.get('test')
    
    ## Set GPU mode 
    device = check_gpu(gpu)
    
    # Initialize variables
    test_loss = 0
    accuracy = 0
    
    model.to(device)

    # Turn off gradients for faster preformance
    with torch.no_grad():
        for inputs, labels in testloader:
        
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss_test = criterion(logps, labels)
                    
            test_loss += batch_loss_test.item()
                    
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader)*100:.3f} %")
   

def predict(image_path, model, gpu = 'off', topk=5):
    
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    Function returns the top k probabilities.
        
    Parameters:
     image_path        - full path to the image which should be classified by the model
     model             - a pre-trained deep learning model, hast to include image 
                         input size
     gpu               - GPU-mode "on" or "off"
     topk              - to classes to be returned by function  
      
    Returns:
     probability_top_k - variable with the top k probabilities infered by the model 
                         from the image  
    '''
    
    # Check if GPU mode is available or not
    device = check_gpu(gpu)         
        
    # Put model in evaluation mode instead of training mode (= default)
    model = model.eval()
    
    # Send model to available device (either 'gpu' or 'cuda')
    model.to(device)
    
    # Open the image from image path
    img = Image.open(image_path)
    
    # Pre-process the image (includes cropping to proper input size saved in model 
    # checkpoint data)
    img_torch = process_images(img, model.input_size)
    
    # Batch dimension: Add dim 0 to preprocessed image so it can be fed to the model
    img_torch = img_torch.unsqueeze(0)
    
    # Send image data to available device (see check above)      
    img_torch  = img_torch.to(device)
    
    # Turn off gradients to speed up the feed forward step
    with torch.no_grad():
        
        # Forward feed, returns log probabilities (last leyer of classifier = log Softmax)
        logps = model.forward(img_torch)
    
    # Get probabilities from log probabilities
    ps = torch.exp(logps)
    
    # Get the top k probabilities and their class indices
    top_p, top_class = ps.topk(topk)
    
    # Return top probabilities and indices
    return top_p, top_class 
