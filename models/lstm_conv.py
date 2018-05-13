import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers

import metrics
import processing
import utils
from layers.conv_deform_1d import ConvDeform1D, describe_deform_layer


class CustomLSTM(keras.layers.LSTM):
    pass


def create_model(input_shape: tuple,
                 batch_size=1,
                 stateful: bool=True,
                 conv_units: int = 64,
                 conv_blocks: int = 1,
                 use_deform: bool = False,
                 return_sequences: bool=True):

    batch_input_shape = (batch_size, ) + input_shape
    print('Model batch input shape: {0}'.format(batch_input_shape))

    model = keras.models.Sequential()
    model.add(layers.InputLayer(batch_input_shape=batch_input_shape))
    #model.add(layers.BatchNormalization())
    #model.add(layers.LSTM(conv_units, stateful=stateful, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, batch_input_shape=batch_input_shape))
    #model.add(layers.Reshape((input_shape[0], conv_units)))
    model.add(layers.Conv1D(conv_units, kernel_size=3, padding='same', activation='relu'))

    for i in range(conv_blocks):
        if use_deform:
            model.add(ConvDeform1D(conv_units,
                                   kernel_size=2,
                                   kernel_initializer=keras.initializers.RandomNormal(0, 0.001),
                                   dilation_rate=2 ** i))
        model.add(layers.Conv1D(conv_units,
                                kernel_size=2,
                                padding='valid',
                                dilation_rate=2**i,
                                activation='relu'))

    if return_sequences:
        model.add(layers.TimeDistributed(layers.Dense(1, activation='linear')))
    else:
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='linear', use_bias=True))

    model.compile(optimizer='adam', loss='mse')
    return model


def lstm_states(model: keras.models.Sequential):
    lstm_layers = [l for l in model.layers if isinstance(l, keras.layers.LSTM)]
    lstm_layer = lstm_layers[-1]  # type: keras.layers.LSTM
    lstm_layer.states


def main():
    path = 'D:\\ydata-labeled-time-series-anomalies-v1_0\\A4Benchmark\\A4Benchmark-TS3.csv'
    window = 64
    epochs = 100
    stop_loss = 0.001
    batch_size = 64
    is_stateful = True
    return_sequences = False
    window_stride = 1
    use_time_diff = False
    train_test_split = 0.6

    # Raw time-series
    raw_ts = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]
    data = processing.TimeSeries(raw_ts,
                                 window=window,
                                 window_stride=window_stride,
                                 return_sequences=return_sequences,
                                 train_test_split=train_test_split,
                                 use_time_diff=use_time_diff)

    # Create model
    model = create_model(input_shape=data.input_shape,
                         batch_size=batch_size,
                         stateful=is_stateful,
                         return_sequences=return_sequences)
    model.summary()
    describe_deform_layer(model, data.x_train[:batch_size])
    model.reset_states()

    for epoch in range(epochs):
        losses = []
        for t, x_batch, y_batch in data.train_samples_generator(batch_size=batch_size, shuffle=True):
            loss = model.train_on_batch(x_batch, y_batch)
            losses.append(loss)
        epoch_loss = np.mean(losses)
        print('Epoch {0}/{1} Loss: {2:.4f}'.format(epoch, epochs, epoch_loss))
        model.reset_states()
        if epoch_loss < stop_loss:
            break

    describe_deform_layer(model, data.x_train[:batch_size])

    y_true = []
    y_pred = []
    t_all = []
    for t, x_batch, y_batch in data.all_samples_generator(batch_size=batch_size):
        predicted_part = model.predict_on_batch(x_batch)
        y_pred += list(predicted_part)
        y_true += list(y_batch)
        t_all += list(t)

    y_pred = data.inverse_transform_predictions(np.array(t_all), np.array(y_pred))
    y_true = data.inverse_transform_predictions(np.array(t_all), np.array(y_true))

    plt.plot(y_pred, label='Predicted')
    plt.plot(y_true, label='True')
    plt.legend()
    plt.grid()
    plt.show()

    # Compute metrics
    _, test_y = utils.split(y_true, ratio=train_test_split)
    _, test_y_pred = utils.split(y_pred, ratio=train_test_split)
    import pprint
    metric_results = metrics.evaluate_all(test_y, test_y_pred)
    pprint.pprint(metric_results)


if __name__ == '__main__':
    main()
