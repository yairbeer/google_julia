import glob
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from skimage.color import gray2rgb, rgb2gray
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
from skimage import exposure
import skimage.transform as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


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


def img_rescale(img, scale):
    if scale > 1:
        img = tf.rescale(img, scale, clip=True)
        return img
    else:
        tmp_img = np.zeros(img.shape)
        img = tf.rescale(img, scale)
        original_y, original_x = tmp_img.shape
        scaled_y, scaled_x = img.shape
        tmp_img[((original_y - scaled_y) // 2):((original_y + scaled_y) // 2),
                ((original_x - scaled_x) // 2):((original_x + scaled_x) // 2)] = img
        return tmp_img


def img_updown(img, up):
    h = img.shape[0]
    h = int(h * up)
    tmp_img = np.zeros(img.shape)
    return tmp_img


def img_leftright(img, right):
    w = img.shape[1]
    w = int(w * right)
    tmp_img = np.zeros(img.shape)
    return tmp_img


def img_draw_test(im_arr, im_names, n_imgs):
    plt.figure(1)
    n_rows = int(np.sqrt(n_imgs))
    n_cols = n_imgs / n_rows
    for img_i in range(n_imgs):
        plt.subplot(n_cols, n_rows, img_i + 1)
        plt.title(im_names[img_i])
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
def rescale_intensity_each(image, low, high):
    plow, phigh = np.percentile(img_file, (low, high))
    return np.clip(exposure.rescale_intensity(image, in_range=(plow, phigh)), 0, 1)

"""
Vars
"""
submit_name = 'ImageDataGenerator_tst.csv'
debug = False
n_fold = 2
debug_n = 100
train_times = 5
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
    train_files[i, :, :, :] = rescale_intensity_each(img_file, 20, 80)
for i, img_file in enumerate(test_files):
    test_files[i, :, :, :] = rescale_intensity_each(img_file, 20, 80)

if debug:
    img_draw(train_files, train_names, debug_n)

# Find and borders
for i, img_file in enumerate(train_files):
    train_files[i, :, :, :] = sobel_each(img_file)
for i, img_file in enumerate(test_files):
    test_files[i, :, :, :] = sobel_each(img_file)

# Contrast streching
for i, img_file in enumerate(train_files):
    train_files[i, :, :, :] = rescale_intensity_each(img_file, 5, 95)
for i, img_file in enumerate(test_files):
    test_files[i, :, :, :] = rescale_intensity_each(img_file, 5, 95)

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

i_part = 1.0 / n_fold
batch_size = 128
nb_classes = 62
nb_epoch = 20

np.random.seed(7)
cv_prob = np.random.sample(train_files_gray.shape[0])

# input image dimensions
img_rows, img_cols = img_size, img_size
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# convert class vectors to binary class matrices
train_labels_dummy = np_utils.to_categorical(train_labels, nb_classes)

test_results = []
acc = []
for i_fold in range(n_fold):

    test_cv_ind = np.logical_and(i_fold * i_part <= cv_prob, (i_fold + 1) * i_part > cv_prob)
    train_cv_ind = np.logical_not(np.logical_and(i_fold * i_part <= cv_prob, (i_fold + 1) * i_part > cv_prob))
    X_train, y_train = train_files_gray[train_cv_ind, :, :], train_labels[train_cv_ind]
    X_test, y_test = train_files_gray[test_cv_ind, :, :], train_labels[test_cv_ind]

    """
    Compile Model
    """
    # the data, shuffled and split between train and test sets
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    np.random.seed(1007)  # for reproducibility
    model = Sequential()
    """
    inner layers start
    """
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid', input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    """
    inner layers stop
    """
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.03, decay=1e-5, momentum=0.7, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.reset_states()

    X_train_cp = np.array(X_train, copy=True)
    for epoth_i in range(nb_epoch):
        print('epoch %d' % epoth_i)
        rescale_fac = np.random.normal(1, 0.1, X_train_cp.shape[0])
        right_fac = np.random.normal(0, 0.15, X_train_cp.shape[0])
        up_fac = np.random.normal(0, 0.15, X_train_cp.shape[0])
        for img_i in range(X_train_cp.shape[0]):
            X_train_cp[img_i, :, :] = img_rescale(X_train[img_i, :, :], rescale_fac[img_i])
            X_train_cp[img_i, :, :] = img_leftright(X_train_cp[img_i, :, :], right_fac[img_i])
            X_train_cp[img_i, :, :] = img_updown(X_train_cp[img_i, :, :], up_fac[img_i])
        model.train_on_batch(X_train, Y_train, accuracy=True)

    """
    Get accuracy
    """
    score = model.evaluate(X_test, Y_test, verbose=0, show_accuracy=True)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    acc.append(score[1])

    predicted_results = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
    print(label_encoder.inverse_transform(predicted_results))
    print(label_encoder.inverse_transform(y_test))

    unsuccesful_predict = np.logical_not(predicted_results == y_test)
    img_draw_test(X_test[unsuccesful_predict, 0, :, :], label_encoder.inverse_transform(y_test[unsuccesful_predict]),
                  debug_n)
if n_fold > 1:
    print('The accuracy is %.3f' % np.mean(acc))
"""
Solve and submit test
"""

np.random.seed(1007)  # for reproducibility
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# the data, shuffled and split between train and test sets
train_files_gray = train_files_gray.reshape(train_files.shape[0], 1, img_rows, img_cols)
test_files_gray = test_files_gray.reshape(test_files.shape[0], 1, img_rows, img_cols)

# Fit the whole train data
# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_files_gray)

# fit the model on the batches generated by datagen.flow()
model.fit_generator(datagen.flow(train_files_gray, train_labels_dummy, batch_size=batch_size),
                    samples_per_epoch=train_files.shape[0], nb_epoch=int(nb_epoch * 1.2), show_accuracy=True, verbose=1)
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
