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
    _weights : np.ndarray | None = None
    _biases : np.ndarray | None = None
    def __init__(self,units:int,activation_func:TYPE.ACTIVATION_FUNCTION) -> None:
        self._units = units
        self._activation_func = activation_func
        self.init_params()
    def init_params(self) -> None:
        self._weights = np.random.rand(self._units)
        self._biases = np.zeros(self._units)  
    