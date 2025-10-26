#start
#import python modules
#pass
#import package modules
import pytest
import numpy as np
#import local modules
from library.nn import FNN
from library.nn import Layer
from library.utils import ActivationFunctions
#end

def test_fnn_initialization():
    structure = [
        Layer(3, ActivationFunctions.RELU),
        Layer(2, ActivationFunctions.SIGMOID)
    ]
    
    fnn = FNN(layers=structure)
    assert fnn._layers == structure
    