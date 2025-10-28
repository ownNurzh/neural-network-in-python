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
        self._gradients:list = []
        self._deltas:list = []
        self._values:np.ndarray = np.zeros(units,dtype=np.float32)
        self._activation_func:TYPE.ACTIVATION_FUNCTION = activation_func
        self._activation_cache: dict  = {
            "before_activation":None,
            "after_activation":None,
        }
        self.init_params()
    def init_params(self) -> None:
        self._biases:np.ndarray = np.zeros(self._units)
    def activate(self,prev_values:np.ndarray,weights:np.ndarray):
        be_s = []
        for i in range(len(weights)):
            w = weights[i]
                        
            z = np.dot(prev_values,w) + self._biases
            z = np.sum(z)
            be_s.append(z)
        be_s = np.array(be_s)
        self._values = self._activation_func.activate(be_s)
        self._activation_cache["before_activation"] = be_s
        self._activation_cache["after_activation"] = self._values
    def get_derivative_values(self)-> np.ndarray | None:
        r = self._activation_cache["before_activation"]
        return self._activation_func.derivative(r) if r is not None else None
    def update_delta(self,deltas:np.ndarray,prev_values:np.ndarray):
        self._deltas = deltas # for update biases
        self._gradients = np.outer(deltas,prev_values) # for update weights
        
    



