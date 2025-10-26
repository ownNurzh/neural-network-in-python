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
    units = 5
    layer = Layer(units=units,activation_func=ActivationFunctions.RELU)
    assert layer._units == units
    assert layer._activation_func == ActivationFunctions.RELU
    assert layer._biases is not None
    np.testing.assert_array_equal(layer._biases, np.zeros(units))
    
def test_layer_activation():
    units = 2
    layer = Layer(units=units,activation_func=ActivationFunctions.RELU)
    weights = np.array([[0.2, 0.8, -0.5],[1.0, -1.5, 2.0]])
    values = np.array([1.0, -2.0, 3.0])
    layer.activate(prev_values=values,weights=weights)
    assert len(layer._activation_cache) == len(weights)
    print(layer._activation_cache)
