import numpy as np
import pandas as pd
import dicom
import os
import matplotlib.pyplot as plt
import cv2
import math
import tensorflow as tf
from skimage import measure, morphology
import keras
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D

####################### preprocess stuff ###########
### Constants ####
data_dir = 'D:/Data/stage1_patients/'# The folder with all patient folders
labels = pd.read_csv('D:/Data/stage1_labels/stage1_labels.csv', index_col=0) # The labels .csv file
output_dir = 'D:/Data/pre/' # output folder for saving processed data
much_data = [] # use this for preprocessing
patients = os.listdir(data_dir)


IMG_SIZE_PX = 512  #original 512
SLICE_COUNT = 30
n_classes = 2
batch_size = 1
######################
def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    # Yield successive n-sized chunks from l.
    for i in range(0, len(l), n):
        yield l[i:i + n]
def mean(a):
    return sum(a) / len(a)

##################################################################################################
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


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


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


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    #image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


MIN_BOUND = -1000.0
MAX_BOUND = 400.0


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image


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


############################## Preprocessing stuff ################################
test_data = np.empty((batch_size, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1))
label_data = []

i = 0
for num,patient in enumerate(patients):
    if i == batch_size:
        break
#    if num % 100 == 0:
#        print(num)
    try:
#        print('1')
        first_patient = load_scan(data_dir + patient)
#        print('2')
        first_patient_pixels = get_pixels_hu(first_patient)
#        print('3')
        pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
        pix_resampled = segment_lung_mask(pix_resampled, False)
        pix_resampled = normalize(pix_resampled)
        pix_resampled = zero_center(pix_resampled)

#        print('4')

        label = labels.get_value(patient, 'cancer')
        #if label == 1:
        #    label = np.array([0, 1])
        #elif label == 0:
        #    label = np.array([1, 0])

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


np.save(output_dir + 'testdata-{}-{}-{}'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), test_data)
np.save(output_dir + 'labels', label_data)
print('ok')

############################## Preprocessing stuff End ################################

######## fit batch generator ##########
def fit_generator(path):
    while 1:
        with open(path) as f:
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                x1, x2, y = process_line(line)
                yield ({'input_1': x1, 'input_2': x2}, {'output': y})



############################## CNN stuff  ################################
# load data
much_data = np.load(output_dir + 'testdata-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT))  # load data for the cnn here
much_data = much_data.reshape(batch_size, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1)

input_shape = (IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1)

# Create training data
X_train = much_data
Y_train = np.load(output_dir + 'labels.npy')
Y_train = keras.utils.to_categorical(Y_train, 2)

Xtrain_data = much_data[:-100]
Xvalidation_data = much_data[-100:]
Ytrain_data = Y_train[:-100]
Yvalidation_data = Y_train[-100:]

# Make training data generator friendly
"""
TODO

save the x + y training data into a file

formatted_training_data = np.save.... something

"""


########### Network
model = keras.models.Sequential()
# Block 01
model.add(Dense(input_shape=input_shape))
model.add(AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding='same'))
model.add(Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv1'))
model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid', name='Pool1'))
# Block 02
model.add(Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv2'))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='Pool2'))
model.add(Dropout(rate=0.3))
# Block 03
model.add(Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv3A'))
model.add(Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv3B'))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='Pool3'))
model.add(Dropout(rate=0.4))
# Block 04
model.add(Conv3D(512, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv4A'))
model.add(Conv3D(512, kernel_size=(3,3,3), activation='relu', padding='same', name='Conv4B'))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='Pool4'))
model.add(Dropout(rate=0.5))
# Block 05
model.add(Conv3D(64, kernel_size=(2,2,2), activation='relu', padding='same', name='last_64'))

# Compile
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#model.summary()


hist = model.fit_gerator(fit_generator(formatted_training_data), epochs=20, verbose=0, validation_data=(Xvalidation_data, Yvalidation_data))

model.save(output_dir + 'testmodel.h5')

model = load_model(output_dir + 'testmodel.h5')

scores = model.evaluate(X_train, Y_train, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
############################## Training End ################################
