import numpy as np
import argparse
from pprint import pprint
import keras
import datetime
import tensorflow as tf

from models.conv_models import DDC_SEARCH_SPACE, DDC_SEARCH_SPACE_REDUCED_1, dcc_generic
import processing
import metrics
import genetic
import utils

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


GA_POPULATION_SIZE = 30
GA_ITERATIONS = 1000
MAX_EPOCHS = 1000
BATCH_SIZE = 256
WINDOW_STRIDE = 1
TRAIN_TEST_SPLIT = 0.7
PATIENCE = 20
SHUFFLE = True
FITNESS_METRIC = 'umbrae'


PROCESSING_DEFAULTS = {
    'derivatives': (0, False),
    'scaler': 'processing.StandardScaler',
    'window_size': 64,
    'return_sequences': True,
}

PROCESSING_SEARCH_SPACE = {
    'derivatives': [(0, False), (1, False), (2, False), (1, True), (2, True)],
    'scaler': [None, 'processing.StandardScaler', 'processing.MinMaxScaler'],
    'window_size': [30, 64, 128],
    'return_sequences': [False, True],
}

PROCESSING_SEARCH_SPACE_REDUCED_1 = {
    'derivatives': [(0, False), (2, False), (1, True), (2, True)],
    'scaler': ['processing.StandardScaler', 'processing.MinMaxScaler'],
    'window_size': [64, 128],
}


class Trainer(object):
    def __init__(self, path, output_path):
        raw_data = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]
        raw_data = np.expand_dims(raw_data, axis=-1)
        self.raw_data = raw_data
        self.output = output_path

    def fitness(self, **kwargs):
        try:
            config = PROCESSING_DEFAULTS.copy()
            config.update(kwargs)
            keras.backend.clear_session()

            scaler = utils.instantiate(config['scaler'])
            num_derivatives, use_deepcast_derivatives = config['derivatives']
            data = processing.TimeSeries(self.raw_data,
                                         window=config['window_size'],
                                         window_stride=WINDOW_STRIDE,
                                         return_sequences=config['return_sequences'],
                                         train_test_split=TRAIN_TEST_SPLIT,
                                         num_derivatives=num_derivatives,
                                         deepcast_derivatives=use_deepcast_derivatives,
                                         scaler=scaler,
                                         use_time_diff=False)

            model = dcc_generic(data.input_shape, **config)
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           min_delta=0,
                                                           patience=PATIENCE,
                                                           verbose=0,
                                                           mode='auto')
            history = model.fit(data.x_train, data.y_train,
                                batch_size=BATCH_SIZE,
                                epochs=MAX_EPOCHS,
                                shuffle=True,
                                verbose=2,
                                callbacks=[early_stopping, keras.callbacks.TerminateOnNaN()],
                                validation_split=0.2)

            # Evaluation
            y_true = data.y_test
            y_pred = model.predict(data.x_test, batch_size=BATCH_SIZE)
            y_true = data.inverse_transform_predictions(data.t_test, y_true)[:, 0]
            y_pred = data.inverse_transform_predictions(data.t_test, y_pred)[:, 0]

            scores = metrics.evaluate(y_true, y_pred, metrics=['mse', 'mae', 'umbrae', 'mape'])
            print('Scores on test data: {0}'.format(scores))

            f = -scores['umbrae']

            output_info = kwargs.copy()
            output_info.update({
                'fitness': f,
                'finished': datetime.datetime.now(),
                'epochs_trained': history.epoch[-1],
            })
            output_info.update(scores)

            utils.write_row_to_csv(self.output, **output_info)

            if np.isnan(f):
                return -100

            return f
        except Exception as err:
            print(err)
            return -100


def main(args):
    #search_space = DDC_SEARCH_SPACE.copy()
    #search_space.update(PROCESSING_SEARCH_SPACE)
    search_space = DDC_SEARCH_SPACE_REDUCED_1.copy()
    search_space.update(PROCESSING_SEARCH_SPACE_REDUCED_1)
    pprint(search_space)

    trainer = Trainer(args.path, args.output)
    ga = genetic.GeneticAlgorithm(search_space, trainer.fitness)
    ga.run(n_iters=GA_ITERATIONS, population_size=GA_POPULATION_SIZE)
    print('============================================')
    print('Fitness hits: {0}'.format(ga.fitness_hits))
    print('Cache hits: {0}'.format(ga.cache_hits))
    print('============== HALL OF FAME ================')
    for cfg, f in ga.hall_of_fame:
        print('Fitness: {0}'.format(f))
        pprint(cfg)
        print('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to a specific .csv file or to folder containing .csv files')
    parser.add_argument('--output', type=str, help='Output csv file with the results', default='results.csv')
    main(parser.parse_args())
