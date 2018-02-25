import numpy as np
import re

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import Callback
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    X, y = zip(*[[series[i - window_size:i], series[i]] for i in range(window_size, len(series))])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y

# TODO: build an RNN to perform regression on our time series input/output data


def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    return model


# TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    text = re.sub(r'[^A-Za-z .,:;?!]', ' ', text)

    return text.lower()

# TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model


def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    inputs = [text[i - window_size:i] for i in range(window_size, len(text), step_size)]
    outputs = [text[i] for i in range(window_size, len(text), step_size)]

    return inputs, outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss


def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))

    return model


class NBatchLogger(Callback):
    def __init__(self, display=100):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            print("Batch Loss: ", logs.get('loss'))
