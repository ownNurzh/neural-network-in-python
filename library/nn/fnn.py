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
    _layers:list[Layer] = []
    _weight:Weight
    def __init__(self,layers:list[Layer]) -> None:
        self._layers = layers
        self._weight = Weight(layers)
    def forward(self,inp : np.ndarray) -> np.ndarray:
        pass