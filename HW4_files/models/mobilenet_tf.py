from tensorflow.keras.layers import Activation, Conv2D, Dense, AveragePooling2D, Flatten, BatchNormalization, \
    DepthwiseConv2D
from tensorflow.keras.models import Sequential


def MobileNetv1():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
              input_shape=(32, 32, 3), use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(
        1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(
        2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(
        1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(
        2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(
        1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(
        2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(
        1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(
        1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(
        1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(
        1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(
        1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((3, 3), strides=(
        2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(1024, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(DepthwiseConv2D((3, 3), strides=(
        1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))
    model.add(Conv2D(1024, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', use_bias=False))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.1))
    model.add(Activation('relu'))

    model.add(AveragePooling2D((2, 2), strides=(2, 2), name='avg_pool'))

    model.add(Flatten())

    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model
