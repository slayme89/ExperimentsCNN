import numpy as np
import pandas as pd
import dicom
import os
import matplotlib.pyplot as plt
import cv2
import math
import tensorflow as tf
import scipy.ndimage.interpolation
import keras
from skimage import measure, morphology
from numpy import random
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import load_model, Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, Convolution3D, Conv3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation

#################
### Constants ###
#################

# The folder with all patient folders
DATA_DIR = 'D:/Data/stage1_patients/'
# The labels .csv file destination
labels = pd.read_csv('D:/Data/stage1_labels/stage1_labels.csv', index_col=0)
# output folder for saving processed data and network model
OUTPUT_DIR = 'D:/Data/pre/'

much_data = []
patients = os.listdir(DATA_DIR)

IMG_SIZE_PX = 50   #original 512
SLICE_COUNT = 30    #number of slices per patient
BATCH_SIZE = 1    #number of patients

################################
### Pre processing functions ###
################################
""" Chunks """
def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    # Yield successive n-sized chunks from l.
    for i in range(0, len(l), n):
        yield l[i:i + n]

""" Gets the mean value from 'a' """
def mean(a):
    return sum(a) / len(a)

""" Extracts a image from a DICOM-files in form of a pixel array """
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

"""  """
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

""" Crops out the lungs from a image """
def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

""" Resampling data """
def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

""" Normalization of a image """
def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

""" Zero centering a image """
def zero_center(image):
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN
    return image

""" Loads a DT-scan from 1 patient (DICOM-file) """
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

""" batch generator for .fit_generator that yield random data from 'features' and 'labels' of size 'batch_size' """
def batch_generator(features, labels, batch_size):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1))
    batch_labels = np.zeros((batch_size, 2))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index = random.choice(len(features),1)
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
            yield (batch_features, batch_labels)

##############################
### Preprocessing the data ###
##############################

test_data = np.empty((BATCH_SIZE, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1))
label_data = []

i = 0
for num,patient in enumerate(patients):
    if i == BATCH_SIZE:
        break
#    if num % 100 == 0:
#        print(num)
    try:
        first_patient = load_scan(DATA_DIR + patient)
        first_patient_pixels = get_pixels_hu(first_patient)
        pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
        pix_resampled = segment_lung_mask(pix_resampled, False)
        pix_resampled = normalize(pix_resampled)
        pix_resampled = zero_center(pix_resampled)

        label = labels.get_value(patient, 'cancer')
        slices = [cv2.resize(np.array(each_slice), (IMG_SIZE_PX, IMG_SIZE_PX)) for each_slice in pix_resampled]
        new_slices = []
        chunk_sizes = math.ceil(len(slices) / SLICE_COUNT)
        
        for slice_chunk in chunks(slices, chunk_sizes):
            slice_chunk = list(map(mean, zip(*slice_chunk)))
            new_slices.append(slice_chunk)
        if len(new_slices) == SLICE_COUNT - 1:
            new_slices.append(new_slices[-1])
        if len(new_slices) == SLICE_COUNT - 2:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
        if len(new_slices) == SLICE_COUNT + 2:
            new_val = list(map(mean, zip(*[new_slices[SLICE_COUNT - 1], new_slices[SLICE_COUNT], ])))
            del new_slices[SLICE_COUNT]
            new_slices[SLICE_COUNT - 1] = new_val
        if len(new_slices) == SLICE_COUNT + 1:
            new_val = list(map(mean, zip(*[new_slices[SLICE_COUNT - 1], new_slices[SLICE_COUNT], ])))
            del new_slices[SLICE_COUNT]
            new_slices[SLICE_COUNT - 1] = new_val

        pix_resampled = np.array(new_slices)
        #test_data.append([pix_resampled, label])
        label_data.append(label)
        np.append(test_data, pix_resampled)
        i = i + 1
        print(i)
    except KeyError as e:
        print('This is unlabeled data!')

np.save(OUTPUT_DIR + 'testdata-{}-{}-{}'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), test_data)
np.save(OUTPUT_DIR + 'labels', label_data)
print('ok')

####################################
### Convolutional Neural Network ###
####################################

# load the data
much_data = np.load(OUTPUT_DIR + 'testdata-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT))  # load data for the cnn here
much_data = much_data.reshape(BATCH_SIZE, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1)

input_shape = (IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1)

# Create training data
X_train = much_data
Y_train = np.load(OUTPUT_DIR + 'labels.npy')
Y_train = keras.utils.to_categorical(Y_train, 2)

"""
OLD TRAINING DATA STUFF

Xtrain_data = much_data[:-100]
Ytrain_data = Y_train[:-100]
"""

Xvalidation_data = much_data[-BATCH_SIZE/2:]
Yvalidation_data = Y_train[-BATCH_SIZE/2:]

### Building the network (model) ###

model = keras.models.Sequential()
# Block 01
model.add(AveragePooling3D(input_shape=input_shape,pool_size=(2, 1, 1), strides=(2, 1, 1), padding='same', name='AvgPool1'))
model.add(Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv1'))
model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid', name='MaxPool1'))
# Block 02
model.add(Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv2'))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='MaxPool2'))
model.add(Dropout(rate=0.3))
# Block 03
model.add(Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv3A'))
model.add(Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv3B'))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='MaxPool3'))
model.add(Dropout(rate=0.4))
# Block 04
model.add(Conv3D(512, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv4A'))
model.add(Conv3D(512, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv4B'))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='MaxPool4'))
model.add(Dropout(rate=0.5))
# Block 05 (?? TODO ??)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# Softmax Layer
model.add(Dense(2))
model.add(Activation('softmax'))
print('Done Building\n')

# Compile 
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print('Done Compiling\n')

# Fit network
model.fit_generator(batch_generator(X_train, Y_train, 1),steps_per_epoch=1, epochs=1, verbose=0, validation_data=(Xvalidation_data, Yvalidation_data))
print('done fitting\n')

# Save the model
model.save(OUTPUT_DIR + 'CNN_model.h5')
print('model saved to dir: ' + OUTPUT_DIR)

# Load the model
#model = load_model(OUTPUT_DIR + 'testmodel.h5')

# Evaluate
scores = model.evaluate(X_train, Y_train, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])