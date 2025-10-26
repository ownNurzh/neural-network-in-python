#start
#import python modules
#pass
#import package modules
import numpy as np
#import local modules
from library.nn import Layer
#end
class Weight:
    _weights: np.ndarray
    def __init__(self,layers:list[Layer]) -> None:
        self.initialize_weights(layers)
    def initialize_weights(self,layers:list[Layer]) -> None:
        ls = layers
        for i in range(len(ls) - 1):
            new_weights = np.random.rand(ls[i]._units, ls[i + 1]._units)
            self._weights.append(new_weights)
    