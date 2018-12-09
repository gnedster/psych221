from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt

FILE = '000000003480'

def visualize_layer(model, layer, input):
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

    fig.savefig('images/layer.png')

if __name__ == '__main__':
    cnn_model = load_model('models/cnn.h5')
    # plot_model(cnn_model, to_file='images/model.png')

    mfile = loadmat('trainingdata/' + FILE + '_low.mat')
    input = np.expand_dims(np.array(mfile['sensorL']['data'][0][0][0][0][0]), axis=2)

    visualize_layer(cnn_model, cnn_model.layers[1], input)
