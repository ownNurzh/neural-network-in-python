#start
#import python modules
from abc import ABC, abstractmethod
#import package modules
import numpy as np
#import local modules
#pass
#end
class ActivationFunction(ABC):
    @staticmethod
    @abstractmethod
    def activate(x):
        pass
    @staticmethod
    @abstractmethod
    def derivative(x):
        pass

class Relu(ActivationFunction):
    """Rectified Linear Unit"""
    @staticmethod
    def activate(x : np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(0, x),1)
    @staticmethod
    def derivative(x : np.ndarray) -> np.ndarray:
        return np.where(((x > 0)& (x < 1)), 1, 0)

class Sigmoid(ActivationFunction):
    """Sigmoid"""
    @staticmethod
    def activate(x : np.ndarray) -> np.ndarray:
        x_safe = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_safe))
    @staticmethod
    def derivative(x : np.ndarray) -> np.ndarray:
        s = Sigmoid.activate(x)
        return s * (1 - s)

class ActivationFunctions:
    RELU:type[ActivationFunction] = Relu
    SIGMOID:type[ActivationFunction] = Sigmoid
    
class TYPE:
    ACTIVATION_FUNCTION = type[ActivationFunctions]