import os
import argparse
import numpy as np
import tensorflow.contrib.keras as keras
#import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from ga import GeneticAlgorithm
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


def huber_loss(y_true, y_pred, clip_delta=1.0):
    '''
     ' Huber loss.
     ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
     ' https://en.wikipedia.org/wiki/Huber_loss
    '''
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)


def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))


def dense(x):
    return tf.layers.Dense(x, 1)


def create_model(input_shape: tuple, outputs: int, **kwargs):
    keras.backend.clear_session()

    model_arch = kwargs.pop('model_arch')
    loss = kwargs.pop('loss')
    optimizer = kwargs.pop('optimizer')
    padding = kwargs.pop('conv_padding', 'same')
    conv_filters = kwargs.pop('conv_filters')
    conv_filter_size = kwargs.pop('conv_filter_size')
    conv_layers = kwargs.pop('conv_layers')
    lstm_cells = kwargs.pop('lstm_cells')
    lstm_layers = kwargs.pop('lstm_layers')
    use_max_pooling = kwargs.pop('use_max_pooling', False)
    activation = kwargs.pop('activation')

    conv_kwargs = {
        'filters': conv_filters,
        'kernel_size': conv_filter_size,
        'padding': padding,
        'activation': activation
    }

    lstm_kwargs = {
        'units': lstm_cells,
        'return_sequences': True
    }

    model = keras.models.Sequential()
    if model_arch == 'conv->lstm' or model_arch == 'conv':
        model.add(keras.layers.Conv1D(input_shape=input_shape, **conv_kwargs))
        if use_max_pooling:
            model.add(keras.layers.MaxPooling1D())
        for l in range(conv_layers - 1):
            model.add(keras.layers.Conv1D(**conv_kwargs))
        if model_arch == 'conv->lstm':
            for l in range(lstm_layers):
                model.add(keras.layers.LSTM(**lstm_kwargs))

    if model_arch == 'lstm->conv' or model_arch == 'lstm':
        model.add(keras.layers.LSTM(input_shape=input_shape, **lstm_kwargs))
        for l in range(lstm_layers - 1):
            model.add(keras.layers.LSTM(**lstm_kwargs))

        if model_arch == 'lstm_conv':
            for l in range(conv_layers):
                model.add(keras.layers.Conv1D(**conv_kwargs))

    # DIRTY CHEAT/HACK, old keras can't infer sequence len, so shape is unknown!!!
    # So sequence specified explicitly
    model.add(keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, input_shape[0], model.layers[-1].output_shape[-1]])))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(outputs, activation=None))
    model.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae', 'mape'])
    return model


def as_sequences(a: np.ndarray, window: int = 30):
    """
    Converts an array of shape (n, ...) to (n - window, window, ...) by performing sliding window
    Done by using stride tricks in NumPy so it is done internally with O(1) complexity.

    NOTE: first axis should be samples

    :param a: Original array of shape (n, ...)
    :param window: Window size, e.g. sequence length
    :return: Sequenced array (n - window, window, ...)
    """
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0], ) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def scaled_grads(a: np.ndarray, axis: int = 0):
    grads = np.gradient(a, axis=axis)
    return (grads - grads.mean()) / (grads.std() + 1e-10)


class SequenceScaler(object):
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()

    def fit(self, a: np.ndarray):
        x = np.reshape(a, newshape=(-1, a.shape[-1]))
        self.scaler.fit(x)

    def transform(self, a: np.ndarray):
        x = np.reshape(a, newshape=(-1, a.shape[-1]))
        return np.reshape(self.scaler.transform(x), newshape=a.shape)

    def inverse_transform(self, a: np.ndarray):
        x = np.reshape(a, newshape=(-1, a.shape[-1]))
        return np.reshape(self.scaler.inverse_transform(x), newshape=a.shape)


def main_old(arguments):
    data = np.genfromtxt(arguments.path, delimiter=',', skip_header=1)
    t_raw = data[:, 1]
    anomaly = data[:, 2]

    train_split_size = 0.6
    shuffle_split = False
    epochs = 40
    batch_size = 256
    window_size = 32
    scaler_class = preprocessing.StandardScaler
    derivatives_to_use = 3
    num_points_to_predict = 5

    num_sequences = len(t_raw) - window_size + 1 - num_points_to_predict
    t_seq = np.zeros((num_sequences, window_size, derivatives_to_use + 1))
    t_seq_y = np.zeros((num_sequences, num_points_to_predict))
    for i in range(num_sequences):
        t_seq[i, :, 0] = t_raw[i:i + window_size]

        for gi in range(derivatives_to_use):
            t_seq[i, :, gi + 1] = np.gradient(t_seq[i, :, gi])

        for pi in range(num_points_to_predict):
            t_seq_y[i, pi] = t_raw[i + window_size + pi]

    t_train, t_test, t_train_y, t_test_y = train_test_split(t_seq, t_seq_y,
                                                            train_size=train_split_size, shuffle=shuffle_split)

    scaler_x = SequenceScaler()
    scaler_x.fit(t_train)

    scaler_y = scaler_class()
    scaler_y.fit(t_train_y)

    x_all = scaler_x.transform(t_seq)
    x_train = scaler_x.transform(t_train)
    x_test = scaler_x.transform(t_test)

    y_all = scaler_y.transform(t_seq_y)
    y_train = scaler_y.transform(t_train_y)
    y_test = scaler_y.transform(t_test_y)

    # Training
    model = create_model(x_train.shape[1:], num_points_to_predict)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    # Validation
    y_pred_validation = model.predict(x_test)[:, num_points_to_predict-1]
    y_true_validation = t_test_y[:, num_points_to_predict-1]
    mse = mean_squared_error(y_true_validation, y_pred_validation)
    mae = mean_absolute_error(y_true_validation, y_pred_validation)
    print('MSE: {0:.3f}'.format(mse))
    print('MAE: {0:.3f}'.format(mae))

    # Plot all
    y_pred_all = model.predict(x_all)
    y_true_unscaled = t_seq_y[:, num_points_to_predict-1]
    y_pred_unscaled = scaler_y.inverse_transform(y_pred_all)[:, num_points_to_predict-1]

    plt.title('LSTM -> CONV1D, MSE: {0:.2f}, MAE: {1:.2f}'.format(mse, mae))
    plt.plot(y_true_unscaled, label='True')
    plt.plot(y_pred_unscaled, label='Predicted')
    plt.legend()
    plt.show()

    return

    validations = 5

    stats = []
    for validation in range(validations):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)
        model = create_model(x.shape[1:], 1)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=10, verbose=0)
        loss, mse, mae, mape = model.evaluate(x_test, y_test, verbose=0)
        print('MSE: {0:.3f}'.format(mse))
        print('MAE: {0:.3f}'.format(mae))
        print('MAPE: {0:.3f}'.format(mape))

        stats.append((mse, mae, mape))

    print(arguments.path)
    mse, mae, mape = np.mean(stats, axis=0)
    print('Validation: ')
    print('\tMSE: {0:.3f}'.format(mse))
    print('\tMAE: {0:.3f}'.format(mae))
    print('\tMAPE: {0:.3f}'.format(mape))


class Trainer(object):
    def __init__(self, path: str, log_path: str):
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        self.t_raw = data[:, 1]
        self.log_path = log_path

    def fitness(self, **kwargs):
        original_kwargs = kwargs.copy()
        window_size = kwargs.pop('window_size')
        num_points_to_predict = kwargs.pop('num_points_to_predict')
        num_derivatives = kwargs.pop('num_derivatives')
        epochs = kwargs.pop('epochs')
        batch_size = kwargs.pop('batch_size')
        scaler_class = kwargs.pop('scaler_class', preprocessing.StandardScaler)

        train_split_size = 0.6
        shuffle_split = False

        num_sequences = len(self.t_raw) - window_size + 1 - num_points_to_predict
        t_seq = np.zeros((num_sequences, window_size, num_derivatives + 1))
        t_seq_y = np.zeros((num_sequences, num_points_to_predict))
        for i in range(num_sequences):
            t_seq[i, :, 0] = self.t_raw[i:i + window_size]

            for gi in range(num_derivatives):
                t_seq[i, :, gi + 1] = np.gradient(t_seq[i, :, gi])

            for pi in range(num_points_to_predict):
                t_seq_y[i, pi] = self.t_raw[i + window_size + pi]

        t_train, t_test, t_train_y, t_test_y = train_test_split(t_seq, t_seq_y,
                                                                train_size=train_split_size,
                                                                shuffle=shuffle_split,
                                                                random_state=42)

        scaler_x = SequenceScaler()
        scaler_x.fit(t_train)

        scaler_y = scaler_class()
        scaler_y.fit(t_train_y)

        #x_all = scaler_x.transform(t_seq)
        x_train = scaler_x.transform(t_train)
        x_test = scaler_x.transform(t_test)

        #y_all = scaler_y.transform(t_seq_y)
        y_train = scaler_y.transform(t_train_y)
        y_test = scaler_y.transform(t_test_y)

        # Training
        print('\n\nCreating model: \n\t{0}'.format(original_kwargs))
        model = create_model(x_train.shape[1:], num_points_to_predict, **kwargs)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        loss, mse, mae, mape = model.evaluate(x_test, y_test, verbose=0)

        fitness = -mse
        log_str = 'Trained:\n\t{0}\n\tMSE: {1:.3f}\n\tF: {2:.3f}'.format(original_kwargs, mse, fitness)
        print(log_str)

        with open(self.log_path, 'a') as f:
            f.write(log_str)

        return fitness


def main(arguments):
    trainer = Trainer(arguments.path, arguments.log)
    ga = GeneticAlgorithm({
        'model_arch': ['conv->lstm', 'lstm->conv', 'conv', 'lstm'],
        'optimizer': ['adam', 'sgd', 'RMSprop'],
        'conv_filters': [4, 8, 16, 32, 64, 128],
        'conv_layers': [1, 2, 3, 5],
        'conv_filter_size': [3, 5],
        #'use_max_pooling': [False, True],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'lstm_cells': [4, 8, 16, 32, 64, 128],
        'lstm_layers': [1, 2],
        'loss': ['mse', huber_loss_mean],

        'epochs': [10, 20, 30, 40, 50],
        'batch_size': [32, 64, 128, 256],
        'window_size': [5, 10, 20, 30, 40, 50],
        'num_derivatives': [0, 1, 2, 3, 4, 5],
        # 'scaler_class': [preprocessing.StandardScaler, preprocessing.MinMaxScaler, None],
        'num_points_to_predict': [1, 2, 3]
    }, trainer.fitness, hall_of_fame=100)
    ga.run(100, 100)

    with open(arguments.log, 'a') as f:
        f.write('{0}'.format(ga.hall_of_fame))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--log', type=str)
    main(parser.parse_args())
