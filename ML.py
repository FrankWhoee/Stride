from __future__ import print_function
import json as js
import numpy as np
import time
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

example_json = '{' \
                '"data":[' \
                    '{"ts": 10000, "x":0, "y" : 0, "z": 0},' \
                    '{"ts": 10000, "x":0, "y" : 0, "z": 0},' \
                    '{"ts": 10000, "x":0, "y" : 0, "z": 0},' \
                    '{"ts": 10000, "x":0, "y" : 0, "z": 0}' \
                ']' \
               '}'

def convertToNp(json):
    json = js.loads(example_json)
    data = json["data"]
    output = np.zeros((len(data), 3, 4))
    for i, object in enumerate(data):
        output[i] = object["ts"], object["x"], object["y"], object["z"]
        print(output[i])
    return output


x_train = convertToNp(example_json)

# Set up CNN
batch_size = 2
num_classes = 2
epochs = 8


# x_train = train_data.reshape(fn,fl,fw,1)
# x_test = test_data.reshape(fn,fl,fw,1)

# y_train = train_label.copy()
# y_test = test_label.copy()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(str(num_classes) + " classes set.")
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# Dense layers and output
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
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