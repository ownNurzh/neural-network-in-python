#start
#import python modules
#pass
#import package modules
import numpy as np
#import local modules
from .layer import Layer
#end
class FNN :
    _layers:list[Layer] = []
    def __init__(self,layers:list[Layer]) -> None:
        self._layers = layers
    def forward(self,inp : np.ndarray) -> np.ndarray:
        pass