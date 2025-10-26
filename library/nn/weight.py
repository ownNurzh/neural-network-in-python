#start
#import python modules
#pass
#import package modules
import numpy as np
#import local modules
from library.nn import Layer
#end
class Weight:
    def __init__(self,layers:list[Layer]) -> None:
        self.initialize_weights(layers)
    def initialize_weights(self,layers:list[Layer]) -> None:
        ls = layers
        self._weights = []
        
        for i in range(len(ls) - 1):
            n_in = ls[i]._units
            n_out = ls[i + 1]._units
            new_weights = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
            self._weights.append(new_weights)
    