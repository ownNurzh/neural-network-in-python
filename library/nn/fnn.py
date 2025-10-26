#start
#import python modules
#pass
#import package modules
import numpy as np
#import local modules
from library.nn import Layer
from library.nn import Weight
#end
class FNN :
    def __init__(self,layers:list[Layer]) -> None:
        self._layers:list[Layer] = layers
        self._weight:Weight = Weight(layers)
    def forward(self,inp : np.ndarray) -> None:
        self._layers[0]._values = inp
        for i in range(1,len(self._layers)):
            prev_layer = self._layers[i-1]
            curr_layer = self._layers[i]
            weights = self._weight._weights[i-1]
            curr_layer.activate(prev_values=prev_layer._values,weights=weights)
        