#start
#import python modules
#pass
#import package modules
import pytest
import numpy as np
#import local modules
from library.nn import Layer
from library.utils import ActivationFunctions
#end

def test_layer_initialization():
    units = 5;
    layer = Layer(units=units,activation_func=ActivationFunctions.RELU)
    assert layer._units == units
    assert layer._activation_func == ActivationFunctions.RELU
    assert layer._biases is not None
    np.testing.assert_array_equal(layer._biases, np.zeros(units))
    