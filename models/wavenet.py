import keras
import numpy as np
import tensorflow as tf
from keras import layers

import metrics
import processing
import utils
from models.utils import get_loss_by_name
from layers.ddcconv1d_keras import DDCConv1D


def wavenet_block(num_filters: int, kernel_size: int, dilation_rate: int):
    def fn(input_):
        tanh_out = layers.Conv1D(num_filters,
                                 kernel_size=kernel_size,
                                 dilation_rate=dilation_rate,
                                 padding='causal',
                                 use_bias=False,
                                 activation='tanh')(input_)
        sigmoid_out = layers.Conv1D(num_filters,
                                    kernel_size=kernel_size,
                                    dilation_rate=dilation_rate,
                                    padding='causal',
                                    use_bias=False,
                                    activation='sigmoid')(input_)
        merged = layers.multiply([tanh_out, sigmoid_out])

        res_out = layers.Conv1D(num_filters,
                                kernel_size=1,
                                #activation='relu',
                                padding='same',
                                use_bias=False)(merged)

        skip_out = layers.Conv1D(num_filters,
                                 kernel_size=1,
                                 #activation='relu',
                                 padding='same',
                                 use_bias=False)(merged)
        out = layers.add([input_, res_out])
        return out, skip_out
    return fn


def wavenet_block_simple(num_filters: int, kernel_size: int, dilation_rate: int):
    def fn(_input):
        x = layers.Conv1D(num_filters,
                      kernel_size=kernel_size,
                      dilation_rate=dilation_rate,
                      padding='causal',
                      use_bias=False,
                      activation='relu')(_input)
        return x, x
    return fn


def wavenet(input_shape,
            output_channels=1,
            num_filters: int = 16,
            blocks: int = 6,
            use_skip_connections: bool = True,
            return_sequences: bool = True,
            loss='huber'):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(num_filters,
                      kernel_size=2,
                      dilation_rate=1,
                      use_bias=False,
                      #activation='tanh',
                      padding='causal',
                      name='initial_causal_conv')(inputs)
    skip_connections = []
    for i in range(blocks):
        #x, skip = wavenet_block(num_filters, kernel_size=2, dilation_rate=2**i)(x)
        x, skip = wavenet_block_simple(num_filters, kernel_size=2, dilation_rate=2 ** i)(x)
        skip_connections.append(skip)
    if use_skip_connections:
        x = layers.add(skip_connections)
    #x = layers.Activation('relu')(x)
    x = layers.Conv1D(output_channels,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      #activation='relu',
                      use_bias=False)(x)
    x = layers.Conv1D(output_channels,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      use_bias=False,
                      activation=None)(x)
    if not return_sequences:
        take_last = layers.Lambda(lambda tensor: tensor[:, -1])
        x = take_last(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss=get_loss_by_name(loss), optimizer=keras.optimizers.Adam(lr=1e-3), metrics=[keras.metrics.mae])
    model.summary()
    return model


def ddcc(input_shape,
         batch_size,
         output_channels=1,
         num_filters: int = 16,
         blocks: int = 4,
         return_sequences: bool = True):
    inputs = layers.Input(batch_shape=(batch_size, ) + input_shape)
    x = inputs
    #x = layers.BatchNormalization()(inputs)
    x = layers.Conv1D(num_filters,
                      kernel_size=2,
                      dilation_rate=1,
                      use_bias=False,
                      padding='causal',
                      name='initial_causal_conv')(x)
    for i in range(blocks):
        #off = layers.Dense(x, activation='sigmoid')
        #off = layers.Conv1D(num_filters, kernel_size=2, padding='causal', dilation_rate=2 ** i)(x)
        #off = layers.Lambda(lambda _x: -tf.nn.sigmoid(_x))(off)
        off = None
        """
        x = DDCC1D(num_filters,
                   kernel_size=3,
                   #dilation_rate=2**i,
                   offset_mode='SK',
                   offsets=off)(x)
        """
        #x = ConvDeform1D(num_filters, kernel_size=5, padding='causal', dilation_rate=1)(x)
        x = layers.Conv1D(num_filters, kernel_size=2, padding='causal', dilation_rate=2**i)(x)

    x = layers.Conv1D(output_channels,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      use_bias=False,
                      activation=None)(x)
    if not return_sequences:
        # Take last element in the sequence
        x = layers.Lambda(lambda tensor: tensor[:, -1])(x)

        #x = layers.Flatten()(x)
        #x = layers.Dense(1)(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=[keras.metrics.mae])
    model.summary()
    return model


def main():
    path = 'D:\\ydata-labeled-time-series-anomalies-v1_0\\A4Benchmark\\A4Benchmark-TS3.csv'
    #path = 'D:\\ydata-labeled-time-series-anomalies-v1_0\\A1Benchmark\\real_38.csv'
    #path = 'D:\\NAB-master\\data\\realKnownCause\\machine_temperature_system_failure.csv'
    epochs = 10
    batch_size = 16
    return_sequences = False
    train_test_split = 0.6
    window_size = 64
    window_stride = 1

    # Raw time-series
    raw_ts = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]
    #raw_ts = np.sin(np.linspace(0, 100, 1000))
    data = processing.TimeSeries(raw_ts,
                                 window=window_size,
                                 window_stride=window_stride,
                                 return_sequences=return_sequences,
                                 train_test_split=0.6,
                                 scaler=None,
                                 use_time_diff=False)

    #model = wavenet(data.input_shape, return_sequences=return_sequences)
    model = ddcc(data.input_shape, batch_size=batch_size, return_sequences=return_sequences)
    describe_ddcc_layer(model, data.x_train[:batch_size])

    all_len = batch_size * (len(data.x) // batch_size)
    train_len = batch_size * (len(data.x_train) // batch_size)
    test_len = batch_size * (len(data.x_test) // batch_size)

    model.fit(data.x_train[:train_len], data.y_train[:train_len],
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')],
              validation_data=(data.x_test[:test_len], data.y_test[:test_len]))
    describe_ddcc_layer(model, data.x_train[:batch_size])

    # Evaluation
    y_true = data.y_test[:test_len]
    y_pred = model.predict(data.x_test[:test_len], batch_size=batch_size)
    y_true = data.inverse_transform_predictions(data.t_test[:test_len], y_true)[:, 0]
    y_pred = data.inverse_transform_predictions(data.t_test[:test_len], y_pred)[:, 0]

    from pprint import pprint
    scores = metrics.evaluate(y_true, y_pred)
    pprint(scores)

    # Prediction
    y_true = data.y[:all_len]
    y_pred = model.predict(data.x[:all_len], batch_size=batch_size)

    y_true = data.inverse_transform_predictions(data.t[:all_len], y_true)[:, 0]
    y_pred = data.inverse_transform_predictions(data.t[:all_len], y_pred)[:, 0]

    import matplotlib.pyplot as plt
    plt.plot(y_pred, label='Predicted')
    plt.plot(y_true, label='True')
    plt.legend()
    plt.grid()
    plt.show()

    """
    y_free_will = []
    x = data.x[-1]
    for i in range(300):
        prediction = model.predict_on_batch(np.expand_dims(x, axis=0))[0, -1]
        y_free_will.append(prediction[0])
        #x = utils.shift(x, offset=-1, fill_val=np.expand_dims(prediction, axis=0))
        x = np.concatenate((x[:-1], np.expand_dims(prediction, axis=0)))
        
    y_free_will = data.inverse_target_scale(np.array(y_free_will))
    """


    # Compute metrics
    _, test_y = utils.split(y_true, ratio=train_test_split)
    _, test_y_pred = utils.split(y_pred, ratio=train_test_split)
    import pprint
    metric_results = metrics.evaluate_all(test_y, test_y_pred)
    pprint.pprint(metric_results)


if __name__ == '__main__':
    main()
