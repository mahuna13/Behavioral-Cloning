import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, ELU, Dropout, Activation
from keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import math
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# read CSV lines from the driving log
# returns tuples where [0] is csv line, [1] is a path prefix for image location
def load_csv(data_sources):
    lines = []
    for data_source in data_sources:
        with open(data_source + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append((line, data_source + 'IMG/'))
    return lines[1:]

# adds prefix to the image path
def get_image_path(source_path, prefix):
    filename = source_path.split('/')[-1]
    return prefix + filename

# only loads paths for central images, and steering measurement
def unpack_data(csv_lines):
    image_paths = []
    measurements = []
    for (line, prefix) in csv_lines:
        image_paths.append(get_image_path(line[0], prefix))
        measurements.append(float(line[3]))
    return image_paths, measurements

# plots histogram of steering measurements
def plot_measurements(measurements):
    hist, bins = np.histogram(measurements, 30)
    plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width = 0.05)
    plt.show()

# deletes around 70% of the steering angle = 0 data points
def delete_zero_measurements(image_paths, measurements):
    to_remove = []
    for i in range(len(measurements)):
        if measurements[i] == 0 and np.random.rand() > 0.3:
            to_remove.append(i)
    image_paths = np.delete(image_paths, to_remove, axis=0)
    measurements = np.delete(measurements, to_remove)
    return image_paths, measurements

def random_flip(img,measurement):
    flip_prob = np.random.random()
    if flip_prob > 0.5 and abs(measurement) > 2:
        measurement = -1*measurement
        img = cv2.flip(img, 1)
    return img,measurement

def random_shift(image,measurement,left,right):
    if left == right:
        return image, measurement
    
    shift = np.random.randint(left, right)
    image = image[60:140, abs(left) + shift : image.shape[1] - right + shift, :]
    measurement += -shift/(right-left)/3.0
    
    return image, measurement

def traingenerator(features, labels, batch_size):
    batch_features = np.zeros((batch_size, 64, 64, 3))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        i = 0
        while i < batch_size:
            index = random.choice(range(0, len(features)))
            
            measurement = labels[index]
            img = plt.imread(features[index])
            
            img,angle = random_shift(img,measurement, -25, 25)
            
            img, angle = random_flip(img, angle)
            
            batch_features[i] = cv2.resize(img, (64,64))
            batch_labels[i] = angle
            i+=1
                
        yield batch_features, batch_labels
        
def validgenerator(features, labels, batch_size):
    batch_features = np.zeros((batch_size, 64, 64, 3))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        i = 0
        while i < batch_size:
            index = random.choice(range(0, len(features)))
            angle = labels[index]
            img = cv2.imread(features[index])
            
            batch_features[i] = cv2.resize(img[60:140,:], (64,64))
            batch_labels[i] = angle
            i+=1
                
        yield batch_features, batch_labels

# read collected data, this includes:
# 1. udacity data
# 2. Husband driving data (crossed some lines) using PS4 controller
# 3. Me driving data (almost no lines crossed :P) using PS4 controller
# 4. Driving the area around the bridge in both directions multiple times
# 5. Driving the right turn after the bridge perfectly
data_sources = ['../data/data/', 'DrivingData/', 'DrivingData4/', 'Turn1/', 'RightTurn/']
lines = load_csv(data_sources)

# unpack data for driving a whole lap, this is both udacity data and the data I collected on my own
image_paths, measurements = unpack_data(lines)

# delete 70% of the straight driving data
image_paths, measurements = delete_zero_measurements(image_paths, measurements)

X = np.array(image_paths)
y = np.array(measurements)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.05, random_state = 42)

train_generator = traingenerator(X_train, y_train, 200)
valid_generator = validgenerator(X_valid, y_valid, 200)

# drive net model with dropout layers
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,
        input_shape=(64, 64, 3)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu", W_regularizer=l2(0.001)))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu", W_regularizer=l2(0.001)))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu", W_regularizer=l2(0.001)))
model.add(Convolution2D(64, 3, 3, activation="relu", W_regularizer=l2(0.001)))
model.add(Convolution2D(64, 3, 3, activation="relu", W_regularizer=l2(0.001)))
model.add(Flatten())
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse", lr=0.0001)
model.fit_generator(train_generator, samples_per_epoch = 20000, validation_data=valid_generator, nb_epoch=10, nb_val_samples = len(X_valid))

model.save('model.h5')
