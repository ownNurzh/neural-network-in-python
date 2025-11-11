#start
#import python modules
import json
import random
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
    random.shuffle(data)
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


#Split data for testing and training
print("Preparing data for training")
answer = ["setosa","versicolor","virginica"]
all_datas = {}
training_datas = []
testing_datas = []
for el in data:
    spec = el["species"]
    if not spec in all_datas:
        all_datas[spec] = []
    ns = [el[param] for param in el if param != "species"]
    all_datas[spec].append(ns)

train_data_len = 30
for species in answer:
    arr = all_datas[species]
    for i in range(len(arr)):
        structure_data = {
            "input":arr[i],
            "output":[1 if c == species else 0 for c in answer]
        }
        if i < train_data_len:
            training_datas.append(structure_data)
        else:
            testing_datas.append(structure_data)
 
print(f"Len training datas : {len(training_datas)}")
print(f"Len testing datas : {len(testing_datas)}")
random.shuffle(training_datas)
random.shuffle(testing_datas)
#Create nn
structure = [
    Layer(4, ActivationFunctions.RELU),
    Layer(12, ActivationFunctions.LEAKYRELU),
    Layer(3, ActivationFunctions.SOFTMAX)
]
yuutsu = FNN(layers=structure)

#Training yuutsu
print("Start training...")

for epoch in range(50):
    for train in training_datas:
        output = np.array(yuutsu.forward(train["input"]))
        true_out = np.array(train["output"])
        yuutsu.backprop(true_out,learning_rate=0.01)
        answer = np.argmax(output)
        #print("Epoch - ",epoch," True output - ",true_out,", Yuutsu output - ",output,", answer - ",answer)
print("End training")
#Testing yuutsu
print("Start testing...")
answers = []
for test in testing_datas:
    output = np.array(yuutsu.forward(test["input"]))
    true_out = np.array(test["output"])
    answer = np.argmax(output)
    answers.append(answer == np.argmax(true_out))
answers = np.array(answers)
result = np.sum(answers)
print("End testing")
print("Testing result : True answer - ",result,", False answer - ",answers.size - result," Accuracy - ", result / (len(testing_datas) / 100))