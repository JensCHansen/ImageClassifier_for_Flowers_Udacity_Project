#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */utility_functions.py
       
##################################################################################
# PROGRAMMER: Jens Hansen
# DATE CREATED: 06.06.2020                            
# REVISED DATE: 
# PURPOSE: This set of functions can be used to check the code after programming 
#          each functionmy functions, 
#          Copyright note: Code is inspired by Jennifer S., Udacity, first project
#         'Pet Image Labels'

##################################################################################

# Imports print functions that check the lab
from check_program import *

# Imports functions created for this program
from utility_functions import *

import operator


def check_command_line_arguments(in_arg):
    """
    Prints each of the command line arguments passed in as parameter in_arg, 
    Parameters:
     in_arg -data structure that stores the command line arguments object
    Returns:
     Nothing - just prints to console  
    """
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        
        # Get model list
        model_list = create_model_list()
                 
        # Check if existing model was chosen and display model list otherwise
        
        # If model exists
        if in_arg.arch in model_list:
            print('\nModel architecture "{}" selected'.format(in_arg.arch))
            
            arch_accepted = True
            
            # prints command line agrs
            print("Current Command Line Arguments for Classifier:",
                  "\n    data_dir =",in_arg.data_directory, 
                  "\n    save_dir =",in_arg.save_dir, 
                  "\n    batch size =",in_arg.batch_size,
                  "\n    arch =", in_arg.arch, 
                  "\n    learning rate =", in_arg.learning_rate,
                  "\n    hidden units =", in_arg.hidden_units,
                  "\n    epochs =", in_arg.epochs,
                  "\n    gpu =", in_arg.gpu
                 )
        
        # If model does not exists display model list
        else:
              
            print('\nModel architecture "{}" not found.'
                  '\nPlese run program again and make sure to choose one of the following architectures:'
                  '\n'
                  .format(in_arg.arch))
            
            arch_accepted = False
        
            # Loop over relevant part of the list which prints out available models
            for idx in range(len(model_list)):
                print('model {:2}: {:>15}'.format(idx, model_list[idx]))
            
            print('\n')
            
    return arch_accepted 

                      
def check_label(label_mapping):
    
    # Print dictionary in order, i.e. from smallest to biggest index
    print('\n')
    
    for i in range(1, len(label_mapping)+1,1):
        print("index {:3}, flower name: {:>25}".format(i, label_mapping[i] ) )
        
    print('\n')
    # Print label_mapping
    #for key, value in label_mapping.items():
    #   print("number {:3}, flower name: {:>25}".format(key, value))

        
def check_in_arguments_predict(in_arg):
    """
    Prints each of the command line arguments of 'predict.py' passed in as parameter in_arg,
    
    Parameters:
     in_arg -data structure that stores the command line arguments object
    Returns:
     Nothing - just prints to console  
    """
    
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        
        # prints command line agrs
        print("\nCurrent Command Line Arguments for Classifier:",
              "\n"
              "\n    Path to image            =", in_arg.path_to_image, 
              "\n    Checkpoint               =", in_arg.checkpoint, 
              "\n    Top kategories displayed =", in_arg.top_k,
              "\n    File with category names =", in_arg.category_names, 
              "\n    GPU mode                 =", in_arg.gpu,
              "\n"
             )
            
    return None


def display_model_architecture(model):
    
    print('The following model was loaded:\n{}.'.format(model))
    
    return None


def print_results(results, class_name):
    
    ## First print the infered class and its prpability:
    print('\n')
    
    prediction = max(results.items(), key=operator.itemgetter(1))
    
    print( "The following was predicted with the highest probability:",
           "\nTop 1: name: {:>20}  probability: {:.2f}\n".format( prediction[0], prediction[1] ) )
    
    if prediction[0] == class_name:
        print("Correct class detected")
        
    else:
        print("Wrong class detected.\nTrue class is '{}'.".format(class_name) )
    
    
    ## Show real label
    #class_names = get_class_name(keys, name_mapping)
    
    ## Then print the top k classes and their probabilities:
    
    print( "\nThese are the top {} classes: \n".format( len(results) ) )
    
    count = 0
    
    # for loop to iterate through the dictionary
    for key, value in results.items():
        count += 1
        print( "Top {}: name: {:>20}  probability: {:.2f}".format(count, key, value ) )
    
    print('\n')
    
    return None
 
    
    