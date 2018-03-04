#import tensorflow.contrib.keras as keras
import keras
import tensorflow as tf


def huber_loss(y_true, y_pred, clip_delta=1.0):
    '''
     ' Huber loss.
     ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
     ' https://en.wikipedia.org/wiki/Huber_loss
    '''
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)


def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))


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
    #model.add(keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, input_shape[0], model.layers[-1].output_shape[-1]])))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(outputs, activation=None))
    model.compile(optimizer=optimizer, loss=loss)
    #model.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae', 'mape'])
    return model
