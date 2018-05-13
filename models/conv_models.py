import keras
import logging
from keras import layers as klayers
from models.utils import huber_loss, get_loss_by_name, huber_dodges_naive_loss, naive_forecast

logger = logging.getLogger('CONV_MODELS')


def deepcast(input_shape: tuple):
    inputs = klayers.Input(shape=input_shape)
    x = inputs
    x = klayers.Conv1D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = klayers.Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = klayers.Flatten()(x)
    x = klayers.Dense(1, activation=None)(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-3), metrics=[keras.metrics.mae])
    return model


def deepcast_mod(input_shape: tuple, causal: bool=False, dilated: bool=False, seq2seq: bool=False):
    padding = 'causal' if causal else 'same'
    dilation_rate = 2 if dilated else 1

    inputs = klayers.Input(shape=input_shape)
    x = inputs
    x = klayers.Conv1D(32, kernel_size=3, padding=padding, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = klayers.Conv1D(64, kernel_size=3, padding=padding, dilation_rate=dilation_rate, activation='relu', kernel_initializer='glorot_uniform')(x)
    if seq2seq:
        x = klayers.Conv1D(1, kernel_size=1, padding='same', activation=None)(x)
    else:
        x = klayers.Flatten()(x)
        x = klayers.Dense(1, activation=None)(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-3), metrics=[keras.metrics.mae])
    return model


def dilated_causal_conv_model(input_shape: tuple,
                              num_blocks: int = 6,
                              num_filters: int = 8,
                              return_sequences: bool = True,
                              activation=None,
                              output_channels: int = 1):
    sequence_len = input_shape[0]
    kernel_size = 2
    receptive_field = (2 ** (num_blocks - 1)) * (kernel_size - 1) + 1
    print(receptive_field)
    inputs = klayers.Input(shape=input_shape)
    x = klayers.Conv1D(num_filters, kernel_size=2, dilation_rate=1,
                       activation=activation,
                       use_bias=False, padding='causal',
                       name='input_conv')(inputs)
    for i in range(num_blocks):
        x = klayers.Conv1D(num_filters, kernel_size=kernel_size, padding='causal', dilation_rate=2 ** i, activation=activation)(x)
    x = klayers.Conv1D(output_channels, kernel_size=1, padding='same', use_bias=False, activation=activation)(x)
    x = klayers.Conv1D(output_channels, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
    if not return_sequences:
        # Return last one
        x = klayers.Lambda(lambda tensor: tensor[:, -1])(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(lr=1e-4), metrics=[keras.metrics.mae])
    return model


def _wavenet_block(num_filters: int, kernel_size: int, dilation_rate: int):
    def fn(input_):
        tanh_out = klayers.Conv1D(num_filters,
                                  kernel_size=kernel_size,
                                  dilation_rate=dilation_rate,
                                  padding='causal',
                                  use_bias=False,
                                  activation='tanh')(input_)
        sigmoid_out = klayers.Conv1D(num_filters,
                                     kernel_size=kernel_size,
                                     dilation_rate=dilation_rate,
                                     padding='causal',
                                     use_bias=False,
                                     activation='sigmoid')(input_)
        merged = klayers.multiply([tanh_out, sigmoid_out])

        res_out = klayers.Conv1D(num_filters,
                                 kernel_size=1,
                                 # activation='relu',
                                 padding='same',
                                 use_bias=False)(merged)

        skip_out = klayers.Conv1D(num_filters,
                                  kernel_size=1,
                                  # activation='relu',
                                  padding='same',
                                  use_bias=False)(merged)
        out = klayers.add([input_, res_out])
        return out, skip_out

    return fn


def _wavenet_block_simple(num_filters: int, kernel_size: int, dilation_rate: int):
    def fn(_input):
        x = klayers.Conv1D(num_filters,
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
            loss: str = 'huber'):
    inputs = klayers.Input(shape=input_shape)
    x = klayers.Conv1D(num_filters,
                       kernel_size=2,
                       dilation_rate=1,
                       use_bias=False,
                       # activation='tanh',
                       padding='causal',
                       name='initial_causal_conv')(inputs)
    skip_connections = []
    for i in range(blocks):
        # x, skip = wavenet_block(num_filters, kernel_size=2, dilation_rate=2**i)(x)
        x, skip = _wavenet_block_simple(num_filters, kernel_size=2, dilation_rate=2 ** i)(x)
        skip_connections.append(skip)
    if use_skip_connections:
        x = klayers.add(skip_connections)
    # x = layers.Activation('relu')(x)
    x = klayers.Conv1D(output_channels,
                       kernel_size=1,
                       strides=1,
                       padding='same',
                       # activation='relu',
                       use_bias=False)(x)
    x = klayers.Conv1D(output_channels,
                       kernel_size=1,
                       strides=1,
                       padding='same',
                       use_bias=False,
                       activation=None)(x)
    if not return_sequences:
        take_last = klayers.Lambda(lambda tensor: tensor[:, -1])
        x = take_last(x)
    model = keras.models.Model(inputs=inputs, outputs=x)

    if 'dodge' in loss:
        logger.info('USING DODGE LOSS')
        naive_fc = naive_forecast(inputs, return_sequences=return_sequences)
        loss_fn = (lambda y_true, y_pred: huber_dodges_naive_loss(y_true, y_pred, naive_fc))
    else:
        logger.info('GETTING LOSS BY NAME: {0}'.format(loss))
        loss_fn = get_loss_by_name(loss)

    model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(lr=1e-4), metrics=[keras.metrics.mae])
    model.summary()
    return model


DDC_DEFAULTS = {
    'num_blocks': 6,
    'num_in_conv': 1,
    'num_out_conv': 1,
    'merge_blocks': False,
    'return_sequences': True,

    # Input convolution settings
    'in_conv_activation': 'relu',
    'in_conv_bias': False,
    'in_conv_filters': 16,
    'in_conv_kernel_size': 2,
    'in_conv_padding': 'causal',
    'in_conv_weights_init': 'glorot_uniform',

    # In-Block convolution layers settings
    'b_conv_activation': 'relu',
    'b_conv_bias': False,
    'b_conv_filters': 16,
    'b_conv_kernel_size': 2,
    'b_conv_padding': 'causal',
    'b_conv_weights_init': 'glorot_uniform',
    'b_conv_residual': False,
    'b_dilation_base': 2,
    'b_conv_dilation_base': 2,

    # Output convolution layers settings
    'out_conv_activation': 'relu',
    'out_conv_bias': False,
    'out_conv_filters': 16,
    'out_conv_padding': 'same',
    'out_conv_kernel_size': 1,
    'out_conv_weights_init': 'glorot_uniform',

    'optimizer': 'adam',
    'loss': 'mse',
}

DDC_SEARCH_SPACE = {
    # Architecture: Layers / blocks setup
    'num_blocks': [1, 2, 3, 4, 5, 6, 7],
    'num_in_conv': [0, 1, 2],
    'num_out_conv': [0, 1, 2],
    'merge_blocks': [False, True],

    # Input convolution settings
    'in_conv_activation': [None, 'tanh', 'sigmoid', 'relu'],
    'in_conv_bias': [False, True],
    'in_conv_filters': [8, 16, 32],
    'in_conv_kernel_size': [2, 3],
    'in_conv_padding': ['causal', 'same'],
    'in_conv_weights_init': ['glorot_normal', 'glorot_uniform'],

    # In-Block convolution settings
    'b_conv_activation': [None, 'tanh', 'sigmoid', 'relu'],
    'b_conv_bias': [False, True],
    'b_conv_filters': [8, 16, 32],
    'b_conv_kernel_size': [2, 3],
    'b_conv_weights_init': ['glorot_normal', 'glorot_uniform'],
    'b_conv_padding': ['causal', 'same'],
    'b_conv_residual': [False, True],
    'b_conv_dilation_base': [1, 2, 3],

    # Output convolution settings
    'out_conv_activation': [None, 'tanh', 'sigmoid', 'relu'],
    'out_conv_bias': [False, True],
    'out_conv_filters': [8, 16, 32],
    'out_conv_kernel_size': [1, 2, 3],
    'out_conv_padding': ['causal', 'same'],
    'out_conv_weights_init': ['glorot_normal', 'glorot_uniform'],

    # Optimization
    'loss': ['mse', 'mae', 'huber', 'dodge'],
    'optimizer': ['adam', 'rmsprop', 'sgd'],
}

DDC_SEARCH_SPACE_REDUCED_1 = {
    # Architecture: Layers / blocks setup
    'num_blocks': [4, 5, 6, 7],
    'num_in_conv': [0, 1, 2],
    'num_out_conv': [0, 1, 2],
    'merge_blocks': [False, True],

    # Input convolution settings
    'in_conv_activation': [None, 'tanh', 'relu'],
    'in_conv_bias': [False, True],
    'in_conv_filters': [8, 16, 32],
    'in_conv_kernel_size': [2, 3],
    'in_conv_padding': ['causal', 'same'],
    'in_conv_weights_init': ['glorot_normal', 'glorot_uniform'],

    # In-Block convolution settings
     #'b_conv_activation': [None, 'relu'],
    'b_conv_bias': [False, True],
    'b_conv_filters': [8, 16, 32],
    'b_conv_kernel_size': [2, 3],
    'b_conv_weights_init': ['glorot_normal', 'glorot_uniform'],
    'b_conv_padding': ['causal', 'same'],
    'b_conv_residual': [False, True],
    'b_conv_dilation_base': [1, 2, 3],

    # Output convolution settings
    'out_conv_activation': [None, 'tanh', 'sigmoid'],
    'out_conv_bias': [False, True],
    'out_conv_filters': [8, 16, 32],
    'out_conv_kernel_size': [1, 2, 3],
    'out_conv_padding': ['causal', 'same'],
    'out_conv_weights_init': ['glorot_normal', 'glorot_uniform'],

    # Optimization
    'loss': ['mse', 'huber', 'dodge'],
    #'optimizer': ['adam'],
}


def dcc_generic(input_shape: tuple, **kwargs):
    """
    Dilated Causal Convolution Model with lots of parameters
    """
    config = DDC_DEFAULTS.copy()
    config.update(kwargs)

    inputs = klayers.Input(shape=input_shape)
    x = inputs

    for i in range(config['num_in_conv']):
        x = klayers.Conv1D(config['in_conv_filters'],
                           kernel_size=config['in_conv_kernel_size'],
                           activation=config['in_conv_activation'],
                           use_bias=config['in_conv_bias'],
                           padding=config['in_conv_padding'],
                           kernel_initializer=config['in_conv_weights_init'])(x)

    skip_connections = []
    for i in range(config['num_blocks']):
        residual = x
        x = klayers.Conv1D(config['b_conv_filters'],
                           kernel_size=config['b_conv_kernel_size'],
                           activation=config['b_conv_activation'],
                           use_bias=config['b_conv_bias'],
                           padding=config['b_conv_padding'],
                           dilation_rate=2 ** i,
                           kernel_initializer=config['b_conv_weights_init'])(x)
        if config['b_conv_residual'] and i > 0:
            x = klayers.add([x, residual])
        skip_connections.append(x)

    if config['merge_blocks'] and len(skip_connections) > 1:
        x = klayers.add(skip_connections)

    for i in range(config['num_out_conv']):
        x = klayers.Conv1D(config['out_conv_filters'],
                           kernel_size=config['out_conv_kernel_size'],
                           activation=config['out_conv_activation'],
                           use_bias=config['out_conv_bias'],
                           padding=config['out_conv_padding'],
                           kernel_initializer=config['out_conv_weights_init'])(x)

    x = klayers.Conv1D(1, kernel_size=1, activation=None)(x)
    if not config['return_sequences']:
        x = klayers.Lambda(lambda tensor: tensor[:, -1])(x)

    if 'dodge' in config['loss']:
        naive_fc = naive_forecast(inputs, return_sequences=config['return_sequences'])
        loss = (lambda y_true, y_pred: huber_dodges_naive_loss(y_true, y_pred, naive_fc))
        print('USING DODGE LOSS')
    else:
        loss = get_loss_by_name(config['loss'])

    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss=loss, optimizer=config['optimizer'], metrics=[keras.metrics.mae])
    return model
