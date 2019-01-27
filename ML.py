from __future__ import print_function
import json as js
import numpy as np
import time
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt

# andyJson = open("andy1.json", "r").read()
# jamesJson = open("james1.json", "r").read()
#
# andyJsonTest = open("andylargesample.json", "r").read()
# jamesJsonTest = open("jameslargesample.json", "r").read()

andyJson = open("andylargesample.json", "r").read()
jamesJson = open("jameslargesample.json", "r").read()

andyJsonTest = open("andy1.json", "r").read()
jamesJsonTest = open("james1.json", "r").read()

# ValueError: Error when checking target: expected dense_9 to have shape (2,) but got array with shape (1,)


def convertToNp(json,id):
    json = js.loads(json)
    data = json["data"]
    output = np.zeros((int(len(data)/4), 4, 4))
    for time_span in range(len(output)):
        for time in range(len(output[time_span])):
            object = data[time + time_span * 4]
            output[time_span][time] = id, object["x"], object["y"], object["z"]
    return output


andyData = convertToNp(andyJson, 1)
jamesData = convertToNp(jamesJson, 0)
x_train_pre = np.zeros((len(andyData) + len(jamesData), 4, 4))
for i in range(len(andyData)):
    x_train_pre[i] = andyData[i]
for i in range(len(jamesData)):
    x_train_pre[i + len(andyData)] = jamesData[i]

np.random.shuffle(x_train_pre)

x_train = np.zeros((len(andyData) + len(jamesData), 4, 3))
y_train = np.zeros(len(andyData) + len(jamesData))
for i in range(len(x_train_pre)):
    for k in range(4):
        x_train[i][k] = x_train_pre[i][k][1],x_train_pre[i][k][2],x_train_pre[i][k][3]
    y_train[i] = x_train_pre[i][k][0]

andyTest = convertToNp(andyJsonTest, 1)
jamesTest = convertToNp(jamesJsonTest, 0)
x_test_pre = np.zeros((len(andyTest) + len(jamesTest), 4, 4))
for i in range(len(andyTest)):
    x_test_pre[i] = andyTest[i]
for i in range(len(jamesTest)):
    x_test_pre[i + len(andyTest)] = jamesTest[i]

np.random.shuffle(x_test_pre)

x_test = np.zeros((len(andyTest) + len(jamesTest), 4, 3))
y_test = np.zeros(len(andyTest) + len(jamesTest))
for i in range(len(x_test_pre)):
    for k in range(4):
        x_test[i][k] = x_test_pre[i][k][1],x_test_pre[i][k][2],x_test_pre[i][k][3]
    y_test[i] = x_test_pre[i][k][0]
print(x_test)
print("------------------------------------------------------------")
print(x_train)

# Set up CNN
batch_size = 2
num_classes = 2
epochs = 7

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y train shape:', y_train.shape)
print('y test shape:', y_test.shape)

print(x_train_pre.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(str(num_classes) + " classes set.")
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# Dense layers and output
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
finish_time = str(time.time())
model.save("model"+finish_time[:finish_time.find(".")]+".hf")
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()