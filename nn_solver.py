import glob
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from skimage.color import gray2rgb, rgb2gray
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
from skimage import exposure
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


def img_draw(im_arr, im_names, n_imgs):
    plt.figure(1)
    n_rows = int(np.sqrt(n_imgs))
    n_cols = n_imgs / n_rows
    for img_i in range(n_imgs):
        plt.subplot(n_cols, n_rows, img_i + 1)
        plt.title(im_names[img_i].split('/')[-1].split('.')[0])
        if len(im_arr.shape) == 4:
            img = im_arr[img_i]
        else:
            img = im_arr[img_i]
        plt.imshow(img)
    plt.show()


def imp_img(img_name):
    # read
    img = imread(img_name)
    # if gray convert to color
    if len(img.shape) == 2:
        img = gray2rgb(img)
    return img


@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(each_channel)
def rescale_intensity_each(image):
    plow, phigh = np.percentile(img_file, (10, 90))
    return np.clip(exposure.rescale_intensity(image, in_range=(plow, phigh)), 0, 1)

"""
Vars
"""
submit_name = 'cnn_sobeleach_gray_contrast_fix.csv'
debug = True
debug_n = 64
"""
Import images
"""
img_size = 30

# Train
path = "data"
train_names = sorted(glob.glob(path + "/trainResized/*"))
train_labels_index = pd.DataFrame.from_csv('trainLabels.csv')
train_files = np.zeros((len(train_names), img_size, img_size, 3)).astype('float32')
train_labels = np.zeros((len(train_names),)).astype(str)
for i, name_file in enumerate(train_names):
    image = imp_img(name_file)
    train_files[i, :, :, :] = image
    train_labels[i] = train_labels_index.loc[int(name_file.split('.')[0].split('/')[-1])]['Class']

# Test
test_names = sorted(glob.glob(path + "/testResized/*"))
test_files = np.zeros((len(test_names), img_size, img_size, 3)).astype('float32')
for i, name_file in enumerate(test_names):
    image = imp_img(name_file)
    test_files[i, :, :, :] = image

train_files /= 255
test_files /= 255

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
print(train_files.shape, test_files.shape)
print(np.unique(train_labels))

"""
Image processing
"""
if debug:
    img_draw(train_files, train_names, debug_n)

# Contrast streching
for i, img_file in enumerate(train_files):
    train_files[i, :, :, :] = rescale_intensity_each(img_file)
for i, img_file in enumerate(test_files):
    test_files[i, :, :, :] = rescale_intensity_each(img_file)

if debug:
    img_draw(train_files, train_names, debug_n)

# Find borders
for i, img_file in enumerate(train_files):
    train_files[i, :, :, :] = sobel_each(img_file)
for i, img_file in enumerate(test_files):
    test_files[i, :, :, :] = sobel_each(img_file)

if debug:
    img_draw(train_files, train_names, debug_n)

train_files_gray = np.zeros((len(train_names), img_size, img_size)).astype('float32')
test_files_gray = np.zeros((len(test_names), img_size, img_size)).astype('float32')

# Change to gray
for i, img_file in enumerate(train_files):
    train_files_gray[i, :, :] = rgb2gray(img_file)
for i, img_file in enumerate(test_files):
    test_files_gray[i, :, :] = rgb2gray(img_file)

if debug:
    img_draw(train_files_gray, train_names, debug_n)

# Contrast streching
for i, img_file in enumerate(train_files_gray):
    p0, p100 = np.percentile(img_file, (0, 100))
    train_files_gray[i, :, :] = exposure.rescale_intensity(img_file, in_range=(p0, p100))
for i, img_file in enumerate(test_files_gray):
    p0, p100 = np.percentile(img_file, (0, 100))
    test_files_gray[i, :, :] = exposure.rescale_intensity(img_file, in_range=(p0, p100))

if debug:
    img_draw(train_files_gray, train_names, debug_n)

"""
Configure train/test
"""
np.random.seed(2016)
cv_prob = np.random.sample(train_files.shape[0])
train_cv_ind = cv_prob < 0.75
test_cv_ind = cv_prob >= 0.75
X_train, y_train = train_files_gray[train_cv_ind, :, :], train_labels[train_cv_ind]
X_test, y_test = train_files_gray[test_cv_ind, :, :], train_labels[test_cv_ind]

"""
Compile Model
"""

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 62
nb_epoch = 20

# input image dimensions
img_rows, img_cols = img_size, img_size
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

"""
CV model
"""
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

"""
Get accuracy
"""
predicted_results = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
print(label_encoder.inverse_transform(predicted_results))
print(label_encoder.inverse_transform(y_test))

"""
Solve and submit test
"""
train_labels = np_utils.to_categorical(train_labels, nb_classes)
train_files_gray = train_files_gray.reshape(train_files_gray.shape[0], 1, img_rows, img_cols)
test_files_gray = test_files_gray.reshape(test_files_gray.shape[0], 1, img_rows, img_cols)

# Fit the whole train data
# model.fit(train_files_gray, train_labels, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1)
predicted_results = model.predict_classes(test_files_gray, batch_size=batch_size, verbose=1)
predicted_results = label_encoder.inverse_transform(predicted_results)

test_index = []
for file_name in test_names:
    test_index.append(int(file_name.split('.')[0].split('/')[-1]))

sub_file = pd.DataFrame.from_csv('sampleSubmission.csv')
sub_file.Class = predicted_results
sub_file.index = test_index
sub_file.index.name = 'ID'

sub_file.to_csv(submit_name)

# each_border -> rgb2gray: 0.7018
# each_rescale_intensity -> each_border -> rgb2gray: 0.7081
# each_rescale_intensity -> each_border -> each_rescale_intensity -> rgb2gray:
# Epoch 15, val_loss: 1.2972 - val_acc: 0.7069
# each_rescale_intensity -> each_border -> rgb2gray -> rescale_intensity: Epoch 12, val_loss: 1.2152 - val_acc: 0.7037
# each_equalize_hist -> each_border -> rgb2gray -> equalize_hist: Epoch 16, val_loss: 1.2984 - val_acc: 0.6846
# each_rescale_intensity -> each_border -> rgb2gray -> rescale_intensity, sobel(10, 90):
# Epoch 18, val_loss: 1.2860 - val_acc: 0.7094
