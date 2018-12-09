import keras
import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from os import listdir
from scipy.io import loadmat, savemat

from predict import bilinear_interpolation

# SRCNN
def get_model():
    model = Sequential()
    # upscaling layer
    #
    # model.add(Conv2DTranspose(64,
    #     kernel_size=(8,8),
    #     input_shape=(100,120,1),
    #     strides=(2, 2),
    #     activation='relu',
    #     use_bias=True
    #     ))
    # layers = [Conv2DTranspose(32,
    #     kernel_size=(8,8),
    #     input_shape=(100,120,1),
    #     strides=(2, 2),
    #     activation='relu',
    #     use_bias=True
    #     ), Conv2D(16,
    #     kernel_size=(5,5),
    #     padding='valid',
    #     activation='relu',
    #     use_bias=True,
    #     ), Conv2D(1,
    #     kernel_size=(5,5),
    #     padding='same',
    #     activation='linear',
    #     use_bias=True,
    #     )]

    layers = [Conv2D(64,
        kernel_size=(9,9),
        padding='same',
        activation='relu',
        use_bias=True,
        ), Conv2D(32,
        kernel_size=(3,3),
        padding='same',
        activation='relu',
        use_bias=True,
        ), Conv2D(1,
        kernel_size=(5,5),
        padding='same',
        activation='linear',
        use_bias=True,
        )]

    for layer in layers:
        model.add(layer)

    return (model, layers)

if __name__ == '__main__':
    files = [file.split('_')[0] for file in listdir("trainingdata")]

    X = []
    y = []

    for file in files:
        mfileHigh = loadmat('trainingdata/' + file + '_high.mat')
        mfileMonochrome = loadmat('trainingdata/' + file + '_monochrome.mat')

        X.append(mfileHigh['sensorH']['data'][0][0][0][0][0])
        y.append(mfileMonochrome['sensorM']['data'][0][0][0][0][0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=421)

    model, layers = get_model()
    model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.003),
              metrics=['mean_squared_error'])

    earlystop = EarlyStopping(monitor='mean_squared_error', min_delta=0.00001, patience=5, verbose=1, mode='auto')
    modelcheckpoint = ModelCheckpoint('models/cnn_m.h5', monitor='mean_squared_error', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    history = model.fit(np.expand_dims(np.array(X_train), axis=3), np.expand_dims(np.array(y_train), axis=3), epochs=100, batch_size=32, callbacks=[earlystop, modelcheckpoint], verbose=1)
    model.save('models/cnn_m.h5')

    # layer_to_visualize(model, layers[1], np.array(X_test[0]).shape(1, 100, 120, 1))
