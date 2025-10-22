#start
#import python modules
#pass
#import package modules
import pytest
import numpy as np
#import local modules
from library.utils import ActivationFunctions
#end

@pytest.mark.parametrize("args, expected", [
    (np.array([-1, 0, 1]), np.array([0, 0, 1])),
    (np.array([-5, -2, -0.1]), np.array([0, 0, 0])),
    (np.array([2, 5, 10]), np.array([2, 5, 10])),
])
def test_relu_activation(args,expected):
    result = ActivationFunctions.RELU.activate(args)
    np.testing.assert_array_equal(result, expected)
@pytest.mark.parametrize("args", [
    np.array([-2, 0, 2]),
    np.array([-5, 5]),
    np.array([1, 2, 3]),
])
def test_sigmoid_activation(args):
    expected = 1 / (1 + np.exp(-np.clip(args, -500, 500)))
    result = ActivationFunctions.SIGMOID.activate(args)
    np.testing.assert_allclose(result, expected, rtol=1e-5)