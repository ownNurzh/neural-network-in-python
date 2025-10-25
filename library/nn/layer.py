#start
#import python modules
#pass
#import package modules
import numpy as np
#import local modules
from library.utils import TYPE
#end
class Layer:
    _units : int
    _activation_func : TYPE.ACTIVATION_FUNCTION
    _weights : np.ndarray
    _biases : np.ndarray
    def __init__(self,units:int,activation_func:TYPE.ACTIVATION_FUNCTION) -> None:
        self._units = units
        self._activation_func = activation_func
    def init_params(self) -> None:
        self._weights = None
        self._biases = None
    