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
    """Rectified Linear Unit (clipped)"""
    @staticmethod
    def activate(x : np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(0, x),1)
    @staticmethod
    def derivative(x : np.ndarray) -> np.ndarray:
        return np.where(((x > 0) & (x < 1)), 1, 0)

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

class SoftMax(ActivationFunction):
    """Soft Max"""
    @staticmethod
    def activate(x : np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    @staticmethod
    def derivative(x : np.ndarray) -> np.ndarray:
        s = SoftMax.activate(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    
class LeakyRelu(ActivationFunction):
    alpha = 0.01

    @staticmethod
    def activate(x: np.ndarray) -> np.ndarray:
        return np.minimum(np.where(x > 0, x, LeakyRelu.alpha * x),1)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x < 0, LeakyRelu.alpha, np.where((x > 0) & (x < 1), 1.0, 0.0))


class ActivationFunctions:
    RELU:type[ActivationFunction] = Relu
    SIGMOID:type[ActivationFunction] = Sigmoid
    SOFTMAX:type[ActivationFunction] = SoftMax
    LEAKYRELU:type[ActivationFunction] =LeakyRelu
    
class TYPE:
    ACTIVATION_FUNCTION = type[ActivationFunctions]