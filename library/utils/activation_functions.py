#import python modules
import math
#import package modules
#pass
#import local modules
#pass
class ActivationFunctions:
    @staticmethod
    def relu(x):
        """Rectified linear unit"""
        return max(0,x)
    @staticmethod
    def sigmoid(x):
        """Sigmoid"""
        return 1 / (1 + math.exp(-x))
    @staticmethod
    def tanh(x):
        """Hyperbolic Tangent"""
        return math.tanh(x)