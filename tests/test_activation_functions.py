#start
#import python modules
#pass
#import package modules
import pytest
import numpy as np
#import local modules
from library.utils import ActivationFunctions
#end

def orig_sigmoid(x: np.ndarray) -> np.ndarray:
    x_safe = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_safe))

def orig_softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def orig_softmax_derivative(x: np.ndarray) -> np.ndarray:
    s = orig_softmax(x).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

@pytest.mark.parametrize("args, expected", [
    (np.array([-1, 0, 1]), np.array([0, 0, 1])),
    (np.array([-5, -2, -0.1]), np.array([0, 0, 0])),
    (np.array([2, 5, 10]), np.array([1, 1, 1])),
])
def test_relu_activation(args,expected):
    result = ActivationFunctions.RELU.activate(args)
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize("args", [
    np.array([-2, 0, 2]),
    np.array([-5, 5]),
    np.array([1, 2, 3]),
    np.array([-1000, 1000]),
    np.array([-400, 400]),
])
def test_sigmoid_activation(args):
    expected = orig_sigmoid(args)
    result = ActivationFunctions.SIGMOID.activate(args)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

@pytest.mark.parametrize("args", [
    np.array([0, 0, 0]),
    np.array([1, 1]),
    np.array([0.5, 0.5, 0.2]),
    np.array([0.3,0.3,0.3]),
])
def test_softmax_activation(args):
    expected = orig_softmax(args)
    result = ActivationFunctions.SOFTMAX.activate(args)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

@pytest.mark.parametrize("args, expected", [
    (np.array([0, -1, 2]), np.array([0, 0, 0])),
    (np.array([-5, -2, 0.1]), np.array([0, 0, 1])),
    (np.array([2, 5, 10]), np.array([0, 0, 0])),
])
def test_relu_derivative(args,expected):
    result = ActivationFunctions.RELU.derivative(args)
    np.testing.assert_array_equal(result, expected)
    
@pytest.mark.parametrize("args", [
    np.array([-2, 0, 2]),
    np.array([-5, 5]),
    np.array([1, 2, 3]),
    np.array([-1000, 1000]),
    np.array([-400, 400]),
])
def test_sigmoid_derivative(args):
    expected = orig_sigmoid(args) * (1 - orig_sigmoid(args))
    result = ActivationFunctions.SIGMOID.derivative(args)
    np.testing.assert_allclose(result, expected)

@pytest.mark.parametrize("args", [
    np.array([0, 0, 0]),
    np.array([1, 1]),
    np.array([0.5, 0.5, 0.2]),
    np.array([0.3,0.3,0.3]),
])
def test_softmax_derivative(args):
    expected = orig_softmax_derivative(args)
    result = ActivationFunctions.SOFTMAX.derivative(args)
    np.testing.assert_allclose(result, expected)