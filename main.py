import argparse
import numpy as np
import csv

from genetic import GeneticAlgorithm
from model import create_model, huber_loss_mean
from utils import mean_squared_error, mean_absolute_error, split, StandardScaler, SequenceScaler, MinMaxScaler, NoScaler


def scaled_grads(a: np.ndarray, axis: int = 0):
    grads = np.gradient(a, axis=axis)
    return (grads - grads.mean()) / (grads.std() + 1e-10)


class Trainer(object):
    def __init__(self, path: str, log_path: str):
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        self.t_raw = data[:, 1]

        self.log_path = log_path
        self.log_file = None
        self.log_writer = None

    def fitness(self, **kwargs):
        original_kwargs = kwargs.copy()
        window_size = kwargs.pop('window_size')
        num_points_to_predict = kwargs.pop('num_points_to_predict')
        num_derivatives = kwargs.pop('num_derivatives')
        epochs = kwargs.pop('epochs')
        batch_size = kwargs.pop('batch_size')
        scaler_class = kwargs.pop('scaler_class', StandardScaler)

        train_split_size = 0.6

        num_sequences = len(self.t_raw) - window_size + 1 - num_points_to_predict
        t_seq = np.zeros((num_sequences, window_size, num_derivatives + 1))
        t_seq_y = np.zeros((num_sequences, num_points_to_predict))
        for i in range(num_sequences):
            t_seq[i, :, 0] = self.t_raw[i:i + window_size]

            for gi in range(num_derivatives):
                t_seq[i, :, gi + 1] = np.gradient(t_seq[i, :, gi])

            for pi in range(num_points_to_predict):
                t_seq_y[i, pi] = self.t_raw[i + window_size + pi]

        t_train, t_test = split(t_seq, ratio=train_split_size)
        t_train_y, t_test_y = split(t_seq_y, ratio=train_split_size)

        scaler_x = SequenceScaler(scaler_class)
        scaler_x.fit(t_train)

        scaler_y = scaler_class()
        scaler_y.fit(t_train_y)

        x_train = scaler_x.transform(t_train)
        x_test = scaler_x.transform(t_test)

        y_train = scaler_y.transform(t_train_y)
        #y_test = scaler_y.transform(t_test_y)

        # Training
        print('\n\nCreating model: \n\t{0}'.format(original_kwargs))
        model = create_model(x_train.shape[1:], num_points_to_predict, **kwargs)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Evaluating
        y_pred_raw_output = model.predict(x_test)
        y_pred = scaler_y.inverse_transform(y_pred_raw_output)
        mse = mean_squared_error(t_test_y, y_pred)
        mae = mean_absolute_error(t_test_y, y_pred)

        fitness = -mse

        log_kwargs = original_kwargs.copy()
        log_kwargs.update({'mse': mse, 'mae': mae})

        print('Trained: {0}'.format(log_kwargs))
        print('Fitness: {0:.3f}'.format(fitness))

        if not self.log_writer:
            self.log_file = open(self.log_path, 'a')
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=sorted(log_kwargs.keys()))
            self.log_writer.writeheader()

        if self.log_writer:
            self.log_writer.writerow(log_kwargs)
            self.log_file.flush()

        return fitness

    def __del__(self):
        if self.log_file:
            self.log_file.close()


def main(arguments):
    trainer = Trainer(arguments.path, arguments.log)
    ga = GeneticAlgorithm({
        'model_arch': ['conv->lstm', 'lstm->conv', 'conv', 'lstm'],
        'optimizer': ['adam', 'sgd', 'RMSprop'],
        'conv_filters': [4, 8, 16, 32, 64, 128],
        'conv_layers': [1, 2, 3, 5],
        'conv_filter_size': [3, 5],
        'use_max_pooling': [False, True],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'lstm_cells': [4, 8, 16, 32, 64, 128],
        'lstm_layers': [1, 2],
        'loss': ['mse', huber_loss_mean],

        'epochs': [10, 20, 30, 40, 50],
        'batch_size': [32, 64, 128, 256],
        'window_size': [5, 10, 20, 30, 40, 50],
        'num_derivatives': [0, 1, 2, 3, 4, 5],
        'scaler_class': [StandardScaler, MinMaxScaler, NoScaler],
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
