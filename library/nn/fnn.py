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
    def forward(self,inp : np.ndarray) -> np.ndarray | None:
        if len(inp) != self._layers[0]._units:
            return None
        self._layers[0]._values = inp
        for i in range(1,len(self._layers)):
            prev_layer = self._layers[i-1]
            curr_layer = self._layers[i]
            weights = self._weight._weights[i-1]
            curr_layer.activate(prev_values=prev_layer._values,weights=weights)
        return self._layers[-1]._values
    def backprop(self,true_output:np.ndarray,learning_rate:float):  
        for i in range(len(self._layers) - 1,0,-1):
            curr_layer = self._layers[i]
            delta = None
            if (curr_layer == self._layers[-1]):
                delta = self._layers[-1]._values - true_output
            else:
                derivative_values = curr_layer.get_derivative_values()
                delta = np.dot(self._weight._weights[i].T,self._layers[i+1]._deltas) * derivative_values
            if delta is not None:
                curr_layer.update_delta(deltas=delta,prev_values=self._layers[i-1]._values)
            
        for i in range(len(self._weight._weights)):
            curr_layer = self._layers[i + 1]
            self._weight._weights[i] -= learning_rate * curr_layer._gradients
            curr_layer._biases -= learning_rate * curr_layer._deltas
                
                
        