import glob
import numpy as np
import pandas as pd
from skimage.io import imread, imshow
from sklearn.preprocessing import LabelEncoder
from skimage.color import gray2rgb, rgb2gray
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


def img_draw(im_list, im_names):
    plt.figure(1)
    n_rows = int(np.sqrt(len(im_list)))
    n_cols = len(im_list) / n_rows
    for img_i in range(len(im_list)):
        plt.subplot(n_cols, n_rows, img_i + 1)
        plt.title(im_names[img_i].split('/')[-1].split('.')[0])
        plt.imshow(im_list[img_i])
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

"""
Import images
"""
img_size = 30

# Train
path = "data"
train_names = sorted(glob.glob(path + "/trainResized/*"))
train_labels_index = pd.DataFrame.from_csv('trainLabels.csv')
train_files = np.ones((len(train_names), img_size, img_size, 3))
train_labels = np.ones((len(train_names),)).astype(str)
for i, name_file in enumerate(train_names):
    image = imp_img(name_file)
    train_files[i, :, :, :] = image
    train_labels[i] = train_labels_index.loc[int(name_file.split('.')[0].split('/')[-1])]['Class']

# Test
test_names = sorted(glob.glob(path + "/testResized/*"))
test_files = np.ones((len(test_names), img_size, img_size, 3))
for i, name_file in enumerate(test_names):
    image = imp_img(name_file)
    test_files[i, :, :, :] = image

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
print(train_files.shape, test_files.shape)
print(np.unique(train_labels))

"""
Image processing
"""

# Find borders
for i, img_file in enumerate(train_files):
    train_files[i, :, :, :] = sobel_each(img_file)
for i, img_file in enumerate(test_files):
    test_files[i, :, :, :] = sobel_each(img_file)

train_files_gray = np.ones((len(train_names), img_size, img_size))
test_files_gray = np.ones((len(test_names), img_size, img_size))

# Chane to gray
for i, img_file in enumerate(train_files):
    train_files_gray[i, :, :] = rgb2gray(img_file)
for i, img_file in enumerate(test_files):
    test_files_gray[i, :, :] = rgb2gray(img_file)

"""
Configure train/test
"""

cv_prob = np.random.sample(train_files.shape[0])
train_cv_ind = cv_prob < 0.75
test_cv_ind = cv_prob >= 0.75
X_train, y_train = train_files_gray[train_cv_ind, :, :], train_labels[train_cv_ind]
X_test, y_test = train_files_gray[test_cv_ind, :, :], train_labels[test_cv_ind]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

"""
Compile Model
"""

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 62
nb_epoch = 12

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
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
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
