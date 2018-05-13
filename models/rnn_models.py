import keras
from keras import layers as klayers
from models.utils import huber_loss


def stateless_lstm(input_shape: tuple,
                   return_sequences: bool=True,
                   num_filters=16,
                   lstm_layers: int=2,
                   output_channels: int=1):
    inputs = klayers.Input(shape=input_shape)
    x = inputs
    for i in range(lstm_layers):
        x = klayers.LSTM(num_filters, return_sequences=return_sequences)(x)
    x = klayers.Conv1D(output_channels, kernel_size=1, use_bias=False, padding='same')(inputs)
    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(lr=1e-4), metrics=[keras.metrics.mae])
    model.summary()
    return model
