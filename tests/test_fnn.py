#start
#import python modules
#pass
#import package modules
import pytest
import numpy as np
#import local modules
from library.nn import FNN
from library.nn import Layer
from library.nn import Weight
from library.utils import ActivationFunctions

#end

def test_fnn_initialization():
    structure = [
        Layer(3, ActivationFunctions.RELU),
        Layer(2, ActivationFunctions.SIGMOID)
    ]
    new_weight = Weight(layers=structure)
    fnn = FNN(layers=structure)
    assert fnn._layers == structure
    assert len(fnn._weight._weights) == len(new_weight._weights)
    

def test_fnn_forward():
    structure = [
        Layer(3, ActivationFunctions.RELU),
        Layer(3, ActivationFunctions.RELU),
        Layer(2, ActivationFunctions.SIGMOID)
    ]
    fnn = FNN(layers=structure)
    input_data = np.array([1.0, 0.5, -1.5])
    fnn.forward(inp=input_data)
    assert np.array_equal(fnn._layers[0]._values, input_data)
    #print(fnn._weight._weights)
    #for layer in structure:
        #print(layer._units,layer._values,layer._activation_cache)