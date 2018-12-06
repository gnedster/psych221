import keras
import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from os import listdir
from scipy.io import loadmat, savemat

from predict import bilinear_interpolation

# SRCNN
def get_model():
    model = Sequential()
    # upscaling layer
    # model.add(Conv2DTranspose(64,
    #     kernel_size=(8,8),
    #     input_shape=(100,120,1),
    #     strides=(2, 2),
    #     activation='relu',
    #     use_bias=True
    #     ))

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

def layer_to_visualize(model, layer, input):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    input = np.expand_dims(input, axis=0)

    convolutions = convout1_f(input)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)

    n = convolutions.shape[2]
    n = int(np.ceil(np.sqrt(n)))

    convolutions = np.moveaxis(convolutions, -1, 0)

    # Visualization of each filter of the layer
    fig = plt.figure()
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[i], cmap='viridis')

    fig.savefig('layer1.png')

if __name__ == '__main__':
    files = [file.split('_')[0] for file in listdir("trainingdata")]


    X = []
    Y = []

    for file in files:
        mfileLow = loadmat('trainingdata/' + file + '_low.mat')
        mfileHigh = loadmat('trainingdata/' + file + '_high.mat')

        X.append(bilinear_interpolation(mfileLow['sensorL']['data'][0][0][0][0][0]))
        Y.append(mfileHigh['sensorH']['data'][0][0][0][0][0])

    X_train = np.array(X[1:]).reshape((13, 202, 242, 1))
    X_test = np.array(X[:1]).reshape((1, 202, 242, 1))

    Y_train = np.array(Y[1:]).reshape((13, 202, 242, 1))
    Y_test = np.array(Y[:1]).reshape((1, 202, 242, 1))

    model, layers = get_model()
    model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.003),
              metrics=['mean_squared_error'])

    model.fit(X_train, Y_train, epochs=100, batch_size=4)
    model.save('models/cnn.h5')

    layer_to_visualize(model, layers[1], X_test[0])
