#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */ptrfivt.py
       
##################################################################################
# PROGRAMMER: Jens Hansen
# DATE CREATED: 14.06.2020                            
# REVISED DATE: 
# PURPOSE: Predicts the name of a flower from an image along with the probability
#          of that name and returns the top K most likely classes.
#          The results of the image inference are stored in the dictionary
#          'results_dic'.
#
#          For the basic usage the following inputs from the command line are 
#          needed: 
#          
#          Basic usage:
#          python predict.py /path/to/image checkpoint
#
#          1) /path/to/image                 - specifiy path to the image
#          2) checkpoint                     - checkpoint file with trained 
#                                              deep learning model saved from 
#                                              train.py
#
#          Options:
#          python predict.py /path/to/image checkpoint --option, see below
#
#          1) --top_k N                      - returns the most likely classes N, 
#                                              with N being an integer
#          2) --category_names filename.json - use a mapping of categories to real
#                                              names
#          3) --gpu                          - use GPU inteference
#
##################################################################################

## First Imports python modules

# Imports print functions that check the lab
from check_program import *

# Imports functions created for this program
from utility_functions import *


## Main program function defined below
def main():
        
    ## Run function to get command line inputs from user (as specified above)
    in_arg_predict = get_input_predict()
    
    ## Print the arguments from the command line
    check_in_arguments_predict(in_arg_predict)
    
    ## Load the checkpoint with the saved deep learning model data
    model, criterion = load_model_checkpoint(in_arg_predict.checkpoint)
    
    ## Print the loaded deep learning model architecture
    print('\nThis is the loaded model architecture:', model.arch, '\n')
    
    display_model_architecture(model)
    ## Label mapping:
    #  Get mapping information for the classes
    name_mapping = label_mapping(in_arg_predict.category_names)
     
    # Show name mapping
    check_label(name_mapping)
    
    ## Image Interference:
    #  Print image path
    #print(in_arg_predict.path_to_image) 
    
    #  Get index of the image:
    img_idx = get_index_from_image_path(in_arg_predict.path_to_image)
    
    # Get class of the image
    class_name = get_class_name(img_idx, name_mapping)
    
    # Print class and index of the image
    print( "\nTrue name: {:>10}  corresponding index: {}\n".format(class_name, str(img_idx) ) ) 
    
    ## Predict the class for an image, i.e. get probabilities
    top_p, top_class = predict(in_arg_predict.path_to_image, 
                               model, 
                               in_arg_predict.gpu, 
                               in_arg_predict.top_k)
    
    #  Compile the image inference results with mapping data,
    #  i.e. combine class name with probability value
    results_dic, class_indices = compile_results(top_p, top_class, name_mapping)
    
    #  Print the results
    print_results(results_dic, class_name)
    
    ## Test random batch of images:
    
    # Load the data
    #dataloaders, image_datasets = load_data('ImageClassifier/flowers', 64, model.input_size)
    
    # Test the model
    #test_model(dataloaders, model, in_arg_predict.gpu, criterion)
    
    
## Call to main function to run the program
if __name__ == "__main__":
    main()