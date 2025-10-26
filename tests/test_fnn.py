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
    