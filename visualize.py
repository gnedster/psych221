from keras.models import load_model
from keras.utils import plot_model

if __name__ == '__main__':
    cnn_model = load_model('models/cnn.h5')
    plot_model(cnn_model, to_file='model.png')
