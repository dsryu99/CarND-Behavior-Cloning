import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle

basedir = 'data/'
outfile = 'model.h5'

# read the entries of the driving log
lines = []
with open(basedir + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = lines[1:] # remove the title of the data provided by udacity
print("data size:"+str(len(lines)))
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# generator for the training and validation data set
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                correction = 0.2
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                angle = [center_angle, left_angle, right_angle]
                for i in range(3):
                    name = basedir + 'IMG/' + batch_sample[i].split('\\')[-1]
                    image = cv2.imread(name)
                    images.append(image)
                    angles.append(angle[i])

                    images.append(cv2.flip(image,1))
                    angles.append(angle[i]*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size=32
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
from keras import optimizers
import matplotlib.pyplot as plt

# input_shape=(160,320,3)
# Create the model of NVidia
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))     # Normalization
model.add(Cropping2D(cropping=((70,25),(0,0))))                              # Cropping
model.add(Convolution2D(24, (5, 5), activation='relu'))
model.add(Convolution2D(36, (5, 5), activation='relu'))
model.add(Convolution2D(48, (5, 5), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# lower the learning rate from 0.001 to 0.0005
adam = optimizers.Adam(lr=0.0005)
model.compile(loss='mse', optimizer=adam)
history_object = model.fit_generator(train_generator, steps_per_epoch= \
    len(train_samples)/batch_size, validation_data=validation_generator,
    validation_steps=len(validation_samples)/batch_size, epochs=5)
model.save(outfile)

# plot the progress of training / validation error based on the epoch
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()