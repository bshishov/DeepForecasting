import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.layers as klayers

import time_series as tsutils
import processing
import metrics


class ModelBase(object):
    # Required 'context' information for a model
    input_window = None

    # How many point the model can predict for a single given context
    output_window = None

    # How output is shifted w.r.t. to input window
    offset = 1


class Model(ModelBase):
    def __init__(self,
                 input_shape: tuple = (5, 1),
                 outputs: int = 1):
        self.input_window = input_shape[0]
        self.output_window = outputs
        self.offset = outputs

        model = keras.Sequential()
        model.add(klayers.Conv1D(10, input_shape=input_shape, padding='same', kernel_size=3, activation='relu'))
        model.add(klayers.Conv1D(10, padding='same', kernel_size=3, activation='relu'))
        model.add(klayers.Conv1D(10, padding='same', kernel_size=3, activation='relu'))
        model.add(klayers.Flatten())
        model.add(klayers.Dense(outputs))
        #model.add(klayers.Dense(10, input_shape=input_shape))
        #model.add(klayers.Dense(outputs))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model

    def predict(self, x, *args, **kwargs):
        return self.model.predict(x, *args, **kwargs)

    def train(self, x, y, *args, **kwargs):
        self.model.fit(x, y, *args, **kwargs)


def main():
    path = 'D:\\data\\M3\\M3Other\\N2836.csv'
    data = np.genfromtxt(path)
    print('Data len: {0}'.format(len(data)))
    predict_points = 8

    model = Model()

    ts = tsutils.TimeSeries(data, test_size=predict_points, scaler=processing.StandardScaler())

    x_train, y_train, t_train = ts.train_data(input_window=model.input_window, output_window=model.output_window, expand=True)
    model.train(x_train, y_train, epochs=200)

    #x_test, y_test, t_test = ts.train_data(input_window=model.input_window, output_window=model.output_window)

    ctx = np.expand_dims(ts.get_test_context(model.input_window, expand=True), axis=0)
    y_pred = tsutils.free_run_batch(model.predict, ctx, predict_points, ts, batch_size=1)
    y_true = ts.get_test_data()

    y_pred_flat = ts.inverse_y(np.squeeze(y_pred))
    y_true_flat = ts.inverse_y(np.squeeze(y_true))

    print(metrics.evaluate(y_true_flat, y_pred_flat, metrics=('smape', 'mae', 'umbrae')))

    '''
    x_all, y_all, t_all = ts.train_data(input_window=model.input_window, output_window=model.output_window)
    y_all_pred = model.predict(x_all)

    t_all_flat = ts.inverse_y(np.squeeze(t_all))
    y_all_flat = ts.inverse_y(np.squeeze(y_all))
    y_pred_pred_flat = ts.inverse_y(np.squeeze(y_all_pred))
    plt.plot(t_all_flat, y_all_flat)
    plt.plot(t_all_flat, y_pred_pred_flat)
    plt.show()
    '''


    #y_free_run_flat = np.squeeze(predictions)
    #plt.plot(np.reshape(y_all, (-1, )))
    #plt.plot(np.concatenate((y_pred_flat, y_free_run_flat)))
    #plt.show()


if __name__ == '__main__':
    main()
