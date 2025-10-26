#start
#import python modules
#pass
#import package modules
import numpy as np
#import local modules
from library.utils import TYPE
#end
class Layer:
    def __init__(self,units:int,activation_func:TYPE.ACTIVATION_FUNCTION) -> None:
        self._units:int = units
        self._values:np.ndarray = np.zeros(units,dtype=np.float32)
        self._activation_func:TYPE.ACTIVATION_FUNCTION = activation_func
        self._activation_cache: list[dict[str,np.ndarray]]  = []
        self.init_params()
    def init_params(self) -> None:
        self._biases:np.ndarray = np.zeros(self._units)
    def activate(self,prev_values:np.ndarray,weights:np.ndarray):
        self._activation_cache  = []
        for w in weights:
            z = np.dot(prev_values,w) + self._biases
            a = self._activation_func.activate(z)
            self._activation_cache.append({"before_activation":z,"after_activation":a})
            self._values = a

