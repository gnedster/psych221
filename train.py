import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2DTranspose, Dropout, Flatten
from os import listdir
from scipy.io import loadmat, savemat

def get_model():
    model = Sequential()
    model.add(Conv2DTranspose(1,
        kernel_size=(4,4),
        input_shape=(100,120,1),
        strides=(2, 2),
        activation='relu'
        ))


    return model


if __name__ == '__main__':
    files = [file.split('_')[0] for file in listdir("trainingdata")]

    X = []
    Y = []

    for file in files:
        mfileLow = loadmat('trainingdata/' + file + '_low.mat')
        mfileHigh = loadmat('trainingdata/' + file + '_high.mat')

        X.append(mfileLow['sensorL']['data'][0][0][0][0][0])
        Y.append(mfileHigh['sensorH']['data'][0][0][0][0][0])

    X_train = np.array(X[1:]).reshape((13, 100, 120, 1))
    X_test = np.array(X[:1]).reshape((1, 100, 120, 1))

    Y_train = np.array(Y[1:]).reshape((13, 202, 242, 1))
    Y_test = np.array(Y[:1]).reshape((1, 202, 242, 1))

    model = get_model()
    model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=100, batch_size=4)
    model.save('models/cnn.h5')





