#!python

from scipy.io import loadmat, savemat
from os import listdir
import numpy as np

def get_output(input):
    print('input size: {} x {}'.format(len(input), len(input[0])))
    output = [[0 for col in range(len(input[0])*2 + 2)] for row in range(len(input)*2 + 2)]
    print('output size: {} x {}'.format(len(output), len(output[0])))
    return output

def nearest_neighbor(input):
    output = get_output(input)

    max_row = len(input)
    max_col = len(input[0])

    for row in range(max_row):
        for col in range(max_col):
            m_row = row // 2 * 2 + row
            m_col = col // 2 * 2 + col

            output[m_row][m_col] = input[row][col]

            if row < max_row - 1:
                output[m_row + 2][m_col] = input[row][col]

            if max_col < max_col - 1:
                output[m_row][m_col + 2] = input[row][col]

            if row < max_row - 1 and col < max_col - 1:
                output[m_row + 2][m_col + 2] = input[row][col]

    return output

def bilinear_interpolation(input):
    output = get_output(input)

    max_row = len(input)
    max_col = len(input[0])

    for row in range(max_row):
        for col in range(max_col):
            output[row*2][col*2] = input[row][col]

            if row < max_row - 2:
                output[row*2+1][col*2] = (input[row][col] + input[row+2][col]) / 2

            if col < max_row_col - 2:
                output[row*2][col*2+1] = (input[row][col] + input[row][col+2]) / 2

            if row < max_row - 2 and col < max_col - 2:
                output[row*2+1][col*2+1] = (input[row][col] + input[row+2][col] + input[row][col+2] + input[row+2][col+2]) / 4

    return output

def mean_squared_error(output, target):
    mse = (np.square(output - target)).mean(axis=None)

if __name__ == '__main__':
    files = [file.split('_')[0] for file in listdir("trainingdata")]

    for file in files:
        mfileLow = loadmat('trainingdata/' + file + '_low.mat')
        mfileHigh = loadmat('trainingdata/' + file + '_high.mat')

        input = mfileLow['sensorL']['data'][0][0][0][0][0]
        output = nearest_neighbor(input)
        savemat('output/'+ file +'_nearest.mat', {'volts': np.array(output, dtype=np.float32)})
