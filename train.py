import argparse
import os
import numpy as np
import tensorflow as tf
import keras
import logging
import datetime
import json
import glob

import processing
import metrics
import utils


from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


DEFAULT_CONFIG = {
    'epochs': 1000,
    'batch_size': 64,
    'window_size': 64,
    'window_stride': 1,
    'return_sequences': True,
    'train_test_split': 0.6,
    'num_derivatives': 0,
    'deepcast_derivatives': False,
    'scaler': 0,
    'patience': 20,
    'shuffle': True,
    'metrics': ['mae', 'rmse', 'mape', 'mase', 'mse', 'umbrae'],
    'model_args': {}
}


def read_file(path, args):
    if args.format == 'csv-1':
        return np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]

    if args.format == 'plain':
        return np.genfromtxt(path)
    raise ValueError('Unsupported format: {0}'.format(args.format))


def run_for_single_file(path, args, config):
    logging.info('Opening: {0}'.format(path))

    # Read the time-series from file
    raw_ts = read_file(path, args)
    if raw_ts.ndim == 1:
        raw_ts = np.expand_dims(raw_ts, axis=-1)

    epochs = config['epochs']
    batch_size = config['batch_size']
    scaler = utils.instantiate(config['scaler'])

    data = processing.TimeSeries(raw_ts,
                                 window=config['window_size'],
                                 window_stride=config['window_stride'],
                                 return_sequences=config['return_sequences'],
                                 train_test_split=config['train_test_split'],
                                 num_derivatives=config['num_derivatives'],
                                 deepcast_derivatives=config['deepcast_derivatives'],
                                 scaler=scaler,
                                 use_time_diff=False)

    model = utils.instantiate(config['model'], data.input_shape, **config['model_args'])  # type: keras.Model
    model.summary()

    all_len = batch_size * (len(data.x) // batch_size)
    train_len = batch_size * (len(data.x_train) // batch_size)
    test_len = batch_size * (len(data.x_test) // batch_size)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0,
                                                   patience=20,
                                                   verbose=0,
                                                   mode='auto')
    history = model.fit(data.x_train[:train_len], data.y_train[:train_len],
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        verbose=2,
                        callbacks=[early_stopping],
                        validation_data=(data.x_test[:test_len], data.y_test[:test_len]))

    if args.weights_out:
        model.save_weights(args.weights_out)

    # Evaluation
    y_true = data.y_test[:test_len]
    y_pred = model.predict(data.x_test[:test_len], batch_size=batch_size)
    y_true = data.inverse_transform_predictions(data.t_test[:test_len], y_true)[:, 0]
    y_pred = data.inverse_transform_predictions(data.t_test[:test_len], y_pred)[:, 0]

    scores = metrics.evaluate(y_true, y_pred, metrics=config['metrics'])
    logging.info('Scores on test data: {0}'.format(scores))

    if not args.noplot:
        output_dir = os.path.dirname(args.output)
        filename = '{0}_{1}.csv'.format(args.run, os.path.splitext(os.path.basename(path))[0])
        plot_path = os.path.join(output_dir, filename)

        # Prediction
        y_true = data.y[:all_len]
        y_pred = model.predict(data.x[:all_len], batch_size=batch_size)

        y_true = data.inverse_transform_predictions(data.t[:all_len], y_true)[:, 0]
        y_pred = data.inverse_transform_predictions(data.t[:all_len], y_pred)[:, 0]

        data = np.vstack((y_true, y_pred)).T
        np.savetxt(plot_path, data, delimiter=',', header='True, Predicted', comments='')
    else:
        plot_path = ''

    if args.output:
        results = config.copy()
        results.update({
            'file': path,
            'run': args.run,
            'plot': plot_path,
            'finished': datetime.datetime.now(),
            'epochs_trained': history.epoch[-1],
        })
        results.update(scores)
        utils.write_row_to_csv(args.output, **results)

    keras.backend.clear_session()


def main(args):
    if not args.run:
        args.run = os.path.splitext(os.path.basename(args.config))[0]
    print('Run: {0}'.format(args.run))

    config = DEFAULT_CONFIG
    with open(args.config) as f:
        config.update(json.load(f))
    print(config)

    files = glob.glob(args.path)
    for path in files:
        try:
            run_for_single_file(path, args, config)
        except Exception as err:
            logging.exception(err)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to a specific .csv file or to folder containing .csv files')
    parser.add_argument('--format', type=str, default='csv-1', help='Input data format')
    parser.add_argument('--output', type=str, help='Output csv file with the results', default='results.csv')
    parser.add_argument('--config', type=str, help='Path to .json config')
    parser.add_argument('--run', type=str, help='Run name')
    parser.add_argument('--weights_out', type=str, help='Output weights path')
    parser.add_argument('--noplot', help='Whether or not plot the results', action='store_true')
    main(parser.parse_args())
