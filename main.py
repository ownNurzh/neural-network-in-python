#start
#import python modules
import json
#import package modules
import numpy as np
#import local modules
from library.nn import Layer
from library.nn import FNN
from library.utils import ActivationFunctions
#end

#Import iris data set
print("Importing data set...")
with open("iris_dataset.json", "r", encoding="utf-8") as f:
    
    data = json.load(f)
    data = np.array(data)
    print(f"file data shape: {data.shape}")
