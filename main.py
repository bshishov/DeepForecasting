import argparse
import csv

import numpy as np

import utils
from genetic import GeneticAlgorithm
from models.model import create_model, huber_loss_mean


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
        scaler_class = kwargs.pop('scaler_class', utils.StandardScaler)

        overall_y_true = []
        overall_y_pred = []

        # Training
        print('\n\nCreating model: \n\t{0}'.format(original_kwargs))
        model = create_model((window_size, num_derivatives + 1), num_points_to_predict, **kwargs)
        model.save_weights('initial_weights.h5')

        for train_t, test_t in utils.roll_cv(self.t_raw, folds=4, backtrack_padding=window_size-1):
            train_x, train_y = utils.as_sequences(train_t, window_size, num_derivatives, num_points_to_predict)
            test_x, test_y = utils.as_sequences(test_t, window_size, num_derivatives, num_points_to_predict)

            scaler_x = utils.SequenceScaler(scaler_class)
            scaler_y = scaler_class()

            train_x_scaled = scaler_x.fit_transform(train_x)
            train_y_scaled = scaler_y.fit_transform(train_y)

            test_x_scaled = scaler_x.transform(test_x)
            test_y_scaled = scaler_y.transform(test_y)

            print('Fitting')
            model.fit(train_x_scaled, train_y_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

            pred_y_scaled = model.predict(test_x_scaled)
            pred_y = scaler_y.inverse_transform(pred_y_scaled)

            overall_y_true.append(test_y)
            overall_y_pred.append(pred_y)

            print('Reset weights')
            model.load_weights('initial_weights.h5')

        all_y_true = np.concatenate(overall_y_true)
        all_y_pred = np.concatenate(overall_y_pred)
        mse = utils.mean_squared_error(all_y_true, all_y_pred)
        mae = utils.mean_absolute_error(all_y_true, all_y_pred)

        print('MSE', mse)
        print('MAE', mae)

        fitness = -mse

        import matplotlib.pyplot as plt
        plt.plot(all_y_pred[:, 0], label='Predicted')
        plt.plot(all_y_true[:, 0], label='True')
        plt.grid()
        plt.legend()
        plt.show()

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
        'scaler_class': [utils.StandardScaler, utils.MinMaxScaler, utils.NoScaler],
        'num_points_to_predict': [1, 2, 3]
    }, trainer.fitness, hall_of_fame=100)

    kwargs = ga.get_kwargs(ga.sample())

    kwargs['scaler_class'] = utils.StandardScaler
    kwargs['epochs'] = 10
    kwargs['model_arch'] = 'conv'
    kwargs['num_derivatives'] = 3
    kwargs['window_size'] = 30
    kwargs['epochs'] = 40
    kwargs['loss'] = huber_loss_mean
    kwargs['use_max_pooling'] = False
    kwargs['conv_filters'] = 64
    kwargs['conv_layers'] = 2
    kwargs['optimizer'] = 'adam'
    kwargs['batch_size'] = 128
    kwargs['num_points_to_predict'] = 1
    kwargs['activation'] = 'relu'
    kwargs['conv_filter_size'] = 5

    #trainer.fitness(**kwargs)

    ga.run(100, 100)

    #with open(arguments.log, 'a') as f:
     #   f.write('{0}'.format(ga.hall_of_fame))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--log', type=str)
    main(parser.parse_args())
