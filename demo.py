from __future__ import print_function
import json as js
import numpy as np
import time
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt
from keras.models import load_model

def convertToNp(json,id):
    json = js.loads(json)
    data = json["data"]
    output = np.zeros((int(len(data)/4), 4, 4))
    for time_span in range(len(output)):
        for time in range(len(output[time_span])):
            object = data[time + time_span * 4]
            output[time_span][time] = id, object["x"], object["y"], object["z"]
    return output


andyJson = open("andy1.json", "r").read()
jamesJson = open("james1.json", "r").read()
andyJson = convertToNp(andyJson, 1)
jamesJson = convertToNp(jamesJson, 0)

usingJames = True

if usingJames:
    x_train_pre = np.zeros((len(jamesJson), 4, 4))
    for i in range(len(jamesJson)):
        x_train_pre[i] = jamesJson[i]
    # for i in range(len(jamesData)):
    #     x_train_pre[i + len(andyData)] = jamesData[i]

    x_train = np.zeros((len(jamesJson), 4, 3))
    for i in range(len(x_train_pre)):
        for k in range(4):
            x_train[i][k] = x_train_pre[i][k][1], x_train_pre[i][k][2], x_train_pre[i][k][3]
else:
    x_train_pre = np.zeros((len(andyJson), 4, 4))
    for i in range(len(andyJson)):
        x_train_pre[i] = andyJson[i]
    # for i in range(len(jamesData)):
    #     x_train_pre[i + len(andyData)] = jamesData[i]

    x_train = np.zeros((len(andyJson), 4, 3))
    for i in range(len(x_train_pre)):
        for k in range(4):
            x_train[i][k] = x_train_pre[i][k][1],x_train_pre[i][k][2],x_train_pre[i][k][3]




model = load_model("Model-88.hf")

new_sample = x_train[3].reshape(1,4,3)

if(model.predict(new_sample)[0][0] < 0.50):
    print("Andy was walking!")
else:
    print("Your phone has been stolen!")

# Andy [[0.25224862 0.7477514 ]]
# James [[0.7459176 0.2540824]]