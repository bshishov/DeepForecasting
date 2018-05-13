import keras
import tensorflow as tf
from keras import layers as klayers
from models.utils import get_loss_by_name, huber_dodges_naive_loss, naive_forecast, naive_from_true


class Shift(keras.layers.Layer):
    def __init__(self, shift: int = 1, return_sequences: bool = True, **kwargs):
        self.return_sequences = return_sequences
        self.kernel = None
        self.shift = shift
        super(Shift, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        if self.return_sequences:
            return tf.concat((x[:, :self.shift, :], x[:, :-self.shift, :]), axis=1)
        else:
            return x[:, -1]

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape
        else:
            return input_shape[0], input_shape[2]


def naive_model(input_shape: tuple, return_sequences: bool = True):
    inputs = klayers.Input(shape=input_shape)
    x = Shift(1, return_sequences=return_sequences)(inputs)
    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss='mse', optimizer='adam', metrics=[keras.metrics.mae])
    return model


def dense(input_shape: tuple,
          return_sequences: bool = False,
          use_bias=False,
          loss='mse'):
    inputs = klayers.Input(shape=input_shape)
    x = inputs
    if not return_sequences:
        outputs = 1
    else:
        outputs = input_shape[0]
    x = klayers.Flatten()(x)
    x = klayers.Dense(outputs, activation=None, use_bias=use_bias)(x)
    model = keras.models.Model(inputs=inputs, outputs=x)

    if 'dodge' in loss:
        naive = naive_forecast(inputs, return_sequences=return_sequences)
        loss_fn = (lambda y_true, y_pred: huber_dodges_naive_loss(y_true, y_pred, naive, alpha=1.0, beta=10.0))
        model.compile(loss=loss_fn, optimizer='adam', metrics=[keras.metrics.mae])
    else:
        model.compile(loss=get_loss_by_name(loss), optimizer='adam', metrics=[keras.metrics.mae])

    return model
