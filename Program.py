import numpy as np
import pandas as pd
import dicom
import os
import matplotlib.pyplot as plt
import cv2
import math
import uuid
import csv
import tensorflow as tf
import scipy.ndimage.interpolation
import keras
from skimage import measure, morphology
from numpy import random
from keras.utils import np_utils
from keras.callbacks import History
from keras.optimizers import SGD
from keras.models import load_model, Sequential, Model
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, AveragePooling3D
from multiprocessing import Pool

################
### Options: ###
################

""" File path options: """
# The data file path (all patients folder)
DATA_DIR = 'D:/Data/stage1_patients/'
# The labels.csv file path
labs = pd.read_csv('D:/Data/stage1_labels/stage1_labels.csv', index_col=0)
# Output folder for saving processed data and network model
OUTPUT_DIR = 'D:/Data/pre/'

""" Network options: """
# Crop img to size (Orginal size: 512)
IMG_SIZE_PX = 32
# Number of slices per patient
SLICE_COUNT = 32
# Formatted data dir to include details about the resolution of the data
FORMATTED_DATA_DIR = OUTPUT_DIR + '-{}-{}-{}'.format(IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT)
# Number of patients (Amounth of data in total)
NUM_PATIENTS = 20
# Training percent (float num, 0.0 = 0%    1 = 100%)
TRAIN_SIZE = 0.8
# Batch_size (should be around 32)
BATCH_SIZE = 1
# Save the model?
SAVE_MODEL = True
# Plot the history of the model-fitting
PLOT_HIST = True

""" Pre processing options: """
# Want to pre process? (if false the program will try to load already pre processed data)
PREPROCESS = True
# Segment Lungs (data)?
SEGMENT = True
# Normalize data?
NORMALIZE = True
# Zero center data?
ZERO_CENT = True


""" Number of threads in pool (for preprocessing)"""
POOL_SIZE = 2

################################
### Pre processing functions ###
################################

""" Chunks """
#n-sized chunks from list l
def chunks( l,n ):
    count=0
    for i in range(0, len(l), n):
        if(count < SLICE_COUNT):
            yield l[i:i + n]
            count=count+1


""" Gets the mean value """
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
    # Method of filling the lung structures
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


#############################
### Create newtwork model ###
#############################
def create_network(input_shape):
    model = keras.models.Sequential()
    # Block 01
    model.add(AveragePooling3D(input_shape=input_shape, pool_size=(2, 1, 1), strides=(2, 1, 1), padding='same', name='AvgPool1'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name='Conv1'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='MaxPool1'))
    # Block 02
    model.add(Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name='Conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='MaxPool2'))
    model.add(Dropout(rate=0.3))
    # Block 03
    model.add(Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name='Conv3A'))
    model.add(Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name='Conv3B'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='MaxPool3'))
    model.add(Dropout(rate=0.4))
    # Block 04
    model.add(Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name='Conv4A'))
    model.add(Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name='Conv4B'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='MaxPool4'))
    model.add(Dropout(rate=0.5))
    # Block 05
    model.add(Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name='Conv5'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(2, activation='softmax'))

    return model

###########################
### Pre Processing data ###
###########################
""" Loads and preprocess labels and patient data """
def preprocessing(labels, patients, pre_process=True, train_size=0.7, segment_data=False, normalize_data=False, zero_cent_data=False, pool_size=POOL_SIZE):
    dictionaryA = {}
    dictionaryB = {}
    train_count = round(train_size * NUM_PATIENTS)

    if PREPROCESS:
        
        patientNames = []
        print('Pre Processing data')
        i = 0
        print('{} / {}'.format(i, NUM_PATIENTS))
        
        for num, patient in enumerate(patients):
            if i == NUM_PATIENTS:
                break
            skip_patient = False
            for preprocessed_patient in os.listdir(OUTPUT_DIR):
                if str(preprocessed_patient)[:-4] == '-{}-{}-{}'.format(IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT) + str(patient):
                    skip_patient = True
                    break
            if skip_patient:
                continue
            try:
                label = labels.get_value(patient, 'cancer')
                first_patient = load_scan(DATA_DIR + patient)
                first_patient_pixels = get_pixels_hu(first_patient)
                pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
                if (segment_data):
                    pix_resampled = segment_lung_mask(pix_resampled, False)
                if (normalize_data):
                    pix_resampled = normalize(pix_resampled)
                if (zero_cent_data):
                    pix_resampled = zero_center(pix_resampled)

                slices = [cv2.resize(np.array(each_slice), (IMG_SIZE_PX, IMG_SIZE_PX)) for each_slice in pix_resampled]
                new_slices = []
                chunk_sizes = math.floor(len(slices) / SLICE_COUNT)

                for slice_chunk in chunks(slices, chunk_sizes):
                    slice_chunk = list(map(mean, zip(*slice_chunk)))
                    new_slices.append(slice_chunk)
                if (len(new_slices) != SLICE_COUNT):
                    continue

                pix_resampled = np.array(new_slices)
                np.save(FORMATTED_DATA_DIR + str(patient), pix_resampled)
                dictionaryB[str(patient)] = label
                patientNames.append(str(patient))
                i += 1
                print('{} / {}'.format(i, NUM_PATIENTS))
            except KeyError as e:
                print('This is unlabeled data!')

        random.shuffle(patientNames)
        dictionaryA['train'] = patientNames[:train_count]
        dictionaryA['validation'] = patientNames[train_count:]
        np.save(FORMATTED_DATA_DIR + 'partitionDict', dictionaryA)
        np.save(FORMATTED_DATA_DIR + 'labelsDict', dictionaryB)
    else:
        dictionaryA = np.load(FORMATTED_DATA_DIR + 'partitionDict.npy').item()
        dictionaryB = np.load(FORMATTED_DATA_DIR + 'labelsDict.npy').item()
        #print('Loaded:')
        #print(dictionaryA)
        #print(dictionaryB)

    return dictionaryA, dictionaryB

######### NEW PREPROCESS ##########

""" New preprocessing func """
def new_preprocess(patient):
    try:
        label = labs.get_value(patient, 'cancer')
        first_patient = load_scan(DATA_DIR + patient)
        first_patient_pixels = get_pixels_hu(first_patient)
        pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
        if (SEGMENT):
            pix_resampled = segment_lung_mask(pix_resampled, False)
        if (NORMALIZE):
            pix_resampled = normalize(pix_resampled)
        if (ZERO_CENT):
            pix_resampled = zero_center(pix_resampled)

        slices = [cv2.resize(np.array(each_slice), (IMG_SIZE_PX, IMG_SIZE_PX)) for each_slice in pix_resampled]
        new_slices = []
        chunk_sizes = math.floor(len(slices) / SLICE_COUNT)

        for slice_chunk in chunks(slices, chunk_sizes):
            slice_chunk = list(map(mean, zip(*slice_chunk)))
            new_slices.append(slice_chunk)
        if (len(new_slices) != SLICE_COUNT):
            return 'Error'

        pix_resampled = np.array(new_slices)
        np.save(FORMATTED_DATA_DIR + str(patient), pix_resampled)
        
    except KeyError as e:
        print('This is unlabeled data!')

    return label

######################
### data Generator ###
######################

""" Generates data for the .fit_generator()"""
class DataGenerator(object):
    'Generates data for Keras'

    def __init__(self, dim_x=32, dim_y=32, dim_z=32, batch_size=32, shuffle=True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle


    def generate(self, labels, list_IDs):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)
            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]
                # Generate data
                X, y = self.__data_generation(labels, list_IDs_temp)

                yield X, y


    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes


    def __data_generation(self, labels, list_IDs_temp):
        'Generates data of batch_size samples'  # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z, 1))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            X[i, :, :, :, 0] = np.load(FORMATTED_DATA_DIR + ID + '.npy')
            # Store class
            y[i] = labels[ID]

        return X, sparsify(y)


def sparsify(y):
    'Returns labels in binary NumPy array'
    n_classes = 2
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])


####################################
### Convolutional Neural Network ###
####################################

# Dir of patients
pats = os.listdir(DATA_DIR)

# Some Pyplot stuff..
plt.rcParams['backend'] = "Qt4Agg"

dictionaryA = {}
dictionaryB = {}


## NEW PREPROCESS ##
if PREPROCESS:
    
    train_count = round(TRAIN_SIZE * NUM_PATIENTS)
    patientNames = []
    
    pool = Pool(processes=POOL_SIZE)
    
    for num, pat in enumerate(patients):
         skip_patient = False
         for preprocessed_patient in os.listdir(OUTPUT_DIR):
            if str(preprocessed_patient)[:-4] == '-{}-{}-{}'.format(IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT) + str(patient):
                skip_patient = True
                break
         if skip_patient:
            continue
         for i in pool.imap_unordered(new_preprocess, pat):
             if(i == 'Error'):
                 continue
             dictionaryB[str(pat)] = i
         patientNames.append(str(pat))
            
     
    random.shuffle(patientNames)
    dictionaryA['train'] = patientNames[:train_count]
    dictionaryA['validation'] = patientNames[train_count:]
    np.save(FORMATTED_DATA_DIR + 'partitionDict', dictionaryA)
    np.save(FORMATTED_DATA_DIR + 'labelsDict', dictionaryB)
else:
    dictionaryA = np.load(FORMATTED_DATA_DIR + 'partitionDict.npy').item()
    dictionaryB = np.load(FORMATTED_DATA_DIR + 'labelsDict.npy').item()


partition = dictionaryA
labels = dictionaryB


"""
# Pre process data
partition, labels = preprocessing(labs,
                                  pats,
                                  pre_process=PREPROCESS,
                                  train_size=TRAIN_SIZE,
                                  segment_data=SEGMENT,
                                  normalize_data=NORMALIZE,
                                  zero_cent_data=ZERO_CENT)
                                  
"""

params = {'dim_x': SLICE_COUNT,
          'dim_y': IMG_SIZE_PX,
          'dim_z': IMG_SIZE_PX,
          'batch_size': BATCH_SIZE,
          'shuffle': True}

# Create data generators
training_generator = DataGenerator(**params).generate(labels, partition['train'])
validation_generator = DataGenerator(**params).generate(labels, partition['validation'])

# Build the network model (3D CNN)
model = create_network((SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX, 1))
print('Done Building\n')

# Compile the model
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print('Done Compiling\n')

# Fit network
hist = model.fit_generator(generator=training_generator,
                           steps_per_epoch=len(partition['train']) / 1,
                           epochs=12,
                           validation_data=validation_generator,
                           validation_steps=len(partition['validation']) / 1
                           )
print('done fitting\n')

# Save the model with a uniqe name
strName = str(uuid.uuid4())
if (SAVE_MODEL):
    if (SEGMENT):
        strName += "_seg"
    if (NORMALIZE):
        strName += "_norm"
    if (ZERO_CENT):
        strName += "_zeroc"
    model.save(OUTPUT_DIR + strName + '.h5')
    print('model saved\n')

# summarize history for accuracy
if (PLOT_HIST):
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title(strName + ' : Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title(strName + ' : Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
plt.show()