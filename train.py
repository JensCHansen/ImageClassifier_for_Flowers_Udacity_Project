#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */train_images.py
       
##################################################################################
# PROGRAMMER: Jens Hansen
# DATE CREATED: 06.06.2020                            
# REVISED DATE: 
# PURPOSE: Trains a new neural network on a dataset and save the trained model as 
#          a checkpoint. 
#          Before training the network the user can specify the following 
#          parameters as input from the command line: 
#          1) directory for saving checkpoint,
#          2) architecture of the network, 
#          3) hyperparameters (learning rate, # of hidden layers, # of epochs), 
#          4) GPU usage
#          The program returns the training loss, the validation loss, and the 
#          validation accuracy

##################################################################################

## Imports python modules

# Imports print functions that check the program
from check_program import *

# Utility functions
from utility_functions import *


## Main program function defined below
def main():
   
    # Initialize variable as False (checks if selected architecture for network exists)
    arch_accepted  = False
    
    # Run function to get command line inputs from user (as specified above)
    in_arg = get_input_args()
    
    # Function that checks command line arguments using in_arg 
    arch_accepted  = check_command_line_arguments(in_arg)
    
    if arch_accepted:
        
        # Label mapping
        cat_to_name = label_mapping()
    
        # check label mapping
        # check_label(cat_to_name)
        
        # Flag for feature extracting.
        # When True we only update the reshaped layer params (always)
        feature_extract = True
    
        # Initialize the model
        model_ft = initialize_model(in_arg.arch, in_arg.hidden_units, feature_extract)
        
        # Load the data
        dataloaders, image_datasets = load_data(in_arg.data_directory, 
                                                in_arg.batch_size, model_ft.input_size)
        
        # Train the model
        model_tr, optimizer_tr, criterion_tr = train_model(model_ft, dataloaders, 
                                                           image_datasets, in_arg)
        
        # Save the trained model 
        save_model_checkpoint(image_datasets, model_tr, optimizer_tr, criterion_tr, in_arg)
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()