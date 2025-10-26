#start
#import python modules
#pass
#import package modules
import pytest
import numpy as np
#import local modules
from library.nn import Layer
from library.nn import Weight
from library.utils import ActivationFunctions
#end

def test_weight_initialization():
    structure = [
        Layer(3, ActivationFunctions.RELU),
        Layer(2, ActivationFunctions.SIGMOID)
    ]
    
    weight = Weight(layers=structure)
    assert len(weight._weights) == len(structure) - 1
    for i in range(len(structure) - 1):
        assert weight._weights[i].shape == (structure[i]._units, structure[i + 1]._units)
    print(weight._weights)