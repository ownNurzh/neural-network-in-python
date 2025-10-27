#start
#import python modules
import json
#import package modules
import numpy as np
#import local modules
from library.nn import Layer
from library.nn import FNN
from library.utils import ActivationFunctions
from library.utils import z_score_normalization
#end

#Import iris data set
print("Importing data set...")
with open("iris_dataset.json", "r", encoding="utf-8") as f: 
    data = json.load(f)
    print(f"File data len: {len(data)}")
    print(f"First index data: {data[0]}")

#Normalization data set
print("Normalization...")
datas_for_normalization = ["sepal_length","sepal_width","petal_length","petal_width"]
print("Param for normalization: ",datas_for_normalization)
for d in datas_for_normalization:
    print(f"Normalizing {d} ...")
    values = np.array([float(el[d]) for el in data])
    z_scores = z_score_normalization(values)
    print(f"Before data sum : {np.sum(values)}")
    print(f"After data sum : {np.sum(z_scores)}")
    #update values
    for i in range(len(data)):
        data[i][d] = z_scores[i]
    a = data[0][d]
    print(f"Updated values {d},finish test type - {type(a)}, value - {a} ")


#Create nn
structure = [
    Layer(4, ActivationFunctions.RELU),
    Layer(12, ActivationFunctions.RELU),
    Layer(3, ActivationFunctions.SOFTMAX)
]
yuutsu = None