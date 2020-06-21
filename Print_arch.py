# Imports python modules

# Imports print functions that check the lab
from check_program import *

# Imports functions created for this program
from utility_functions import *

# Imports python modules
import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn

from collections import OrderedDict

    
# Print architecture
arch_list = create_model_list()
print(arch_list)

model = models.vgg19_bn(pretrained=True)

print(model)

for arch in arch_list:
    
    # Load pre-trained model from torchvision.models with attribute saved in 'arch'
    #model = getattr(models, arch)(pretrained=True)
    None
    
    # Print the model architecture
    #print('\nThis is the architecture of {}: \n{}\n '.format(arch, model))
     
    