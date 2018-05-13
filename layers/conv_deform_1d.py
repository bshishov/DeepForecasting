import tensorflow as tf
import numpy as np
import keras


def batch_indices(indices):
    """
    Computes batch indices for tf.gather_nd

    :param indices: indices tensor of shape [batch_size, sequence_len]
    :return: indices of shape [batch_size, sequence_len, 2]  where [:, :, 0] - sample index in batch, [:, :, 1] sample
    """
    input_shape = tf.shape(indices)
    batch_size = input_shape[0]
    sequence_len = input_shape[1]

    # Create a tensor that indexes into the same batch.
    # This is needed for operations like gather_nd to work.
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1))
    b = tf.tile(batch_idx, (1, sequence_len))
    indices = tf.stack([b, indices], axis=2)
    return indices


def sample_interpolate(x, coords):
    """
    Samples input tensor x by coords with interpolation
    :param x: input tensor of shape [batch_size, sequence_len]
    :param coords: batch coordinates of shape [batch_size, sequence_len]
    :return:
    """

    # Left and right indices, e.g. index of 3.65 would be 3 on the left and 4 on the right
    indices_left = tf.cast(tf.floor(coords), tf.int32)
    indices_right = tf.cast(tf.ceil(coords), tf.int32)

    # Calculate interpolation, for index 3.65 interpolation would be0.65
    interpolation = coords - tf.cast(indices_left, tf.float32)

    # Sample both values (on the lef and right)
    vals_left = tf.gather_nd(x, batch_indices(indices_left))
    vals_right = tf.gather_nd(x, batch_indices(indices_right))

    # Return interpolated values
    return vals_left + (vals_right - vals_left) * interpolation


def transform_to_bc_s(x, shape):
    """ (batch_size, sequence_len, channels) -> (batch_size * channels, sequence_len) """

    # (batch_size, sequence_len, channels) -> (batch_size, channels, sequence_len)
    x = tf.transpose(x, [0, 2, 1])

    # (batch_size, channels, sequence_len) -> (batch_size * channels, sequence_len)
    x = tf.reshape(x, (-1, int(shape[1])))
    return x


def transform_to_b_s_c(x, shape):
    """ (batch_size * channels, sequence_len) -> (batch_size, sequence_len, channels) """

    # (batch_size * channels, sequence_len) -> (batch_size, channels, sequence_len)
    x = tf.reshape(x, shape=(-1, int(shape[2]), int(shape[1])))

    # (batch_size, channels, sequence_len) -> (batch_size, sequence_len, channels)
    x = tf.transpose(x, [0, 2, 1])
    return x


class ConvDeform1D(keras.layers.Conv1D):
    """ConvOffset1D
    Convolutional layer responsible for learning the 1D offsets and output the
    deformed feature map using linear interpolation
    Note that this layer does not perform convolution on the deformed feature map.
    """
    def __init__(self, filters: int, padding: str = 'same', *args, **kwargs):
        """Init
        Parameters
        ----------
        filters: int
            Number of channel of the input feature map
        padding: str
            Padding mode. Only same and causal supported
        dilation_rate: int
            Dilation rate (for dilated convolutions) aka skip_rate
        *args:
            Pass to superclass. See Conv1D layer in Keras
        **kwargs:
            Pass to superclass. See Conv1D layer in Keras
        """
        if padding not in ['same', 'causal']:
            raise ValueError('Padding should be either \'same\' or \'causal\'')
        self.is_causal = padding is 'causal'
        self.filters = filters
        self.offsets = None
        super(ConvDeform1D, self).__init__(self.filters, *args,
                                           padding=padding,
                                           use_bias=False,
                                           #activation='tanh',
                                           #kernel_regularizer=keras.regularizers.l1(0.1),
                                           #bias_regularizer=keras.regularizers.l1(0.1),
                                           kernel_initializer=keras.initializers.random_normal(0, 0.0001),
                                           **kwargs)

    def call(self, x, **kwargs):
        """ Returns the deformed featured map """
        x_shape = x.get_shape()

        # Use Conv1D to obtain offsets of shape (batch_size, sequence_len, channels)
        offsets = super(keras.layers.Conv1D, self).call(x)
        if self.is_causal:
            offsets = -tf.nn.sigmoid(offsets) * self.kernel_size

        self.offsets = offsets

        # reshape offsets: (batch_size, sequence_len, channels) -> (batch_size * channels, sequence_len)
        offsets = transform_to_bc_s(offsets, x_shape)

        # reshape inputs x: (batch_size, sequence_len, channels) -> (batch_size * channels, sequence_len)
        x = transform_to_bc_s(x, x_shape)

        # Sample by coords
        base_indices = tf.expand_dims(tf.range(0, int(x_shape[1]), dtype=tf.float32), axis=0)
        x_deform = sample_interpolate(x, base_indices + offsets)

        # reshape x_deform: (batch_size * channels, sequence_len) -> (batch_size, sequence_len, channels)
        x_deform = transform_to_b_s_c(x_deform, x_shape)
        return x_deform

    def compute_output_shape(self, input_shape):
        """Output shape is the same as input shape
        Because this layer does only the deformation part
        """
        return input_shape


def _test_model():
    import keras.layers as layers

    input_shape = (64, 3)

    model = keras.models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())

    model.add(ConvDeform1D(32, kernel_size=3, kernel_initializer=keras.initializers.RandomNormal(0, 0.01)))
    model.add(layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(ConvDeform1D(64, kernel_size=3, kernel_initializer=keras.initializers.RandomNormal(0, 0.01)))
    model.add(layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    data = np.random.random((1000, ) + input_shape)
    labels = np.random.randint(2, size=(1000, 1))

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, labels, epochs=10, batch_size=32)


def _test_simple():
    raw_x = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ])

    raw_offsets = np.array([
        [0, 0, 0, 0],
        [-0.5, -0.5, -0.5, -0.5],
        [1, 1, 1, 1]
    ], dtype=np.float32)

    x = tf.Variable(raw_x, dtype=tf.float32)
    offsets = tf.Variable(raw_offsets, dtype=tf.float32)
    base_indices = tf.expand_dims(tf.range(0, 4, dtype=tf.float32), axis=0)
    coords = offsets + base_indices
    res = sample_interpolate(x, coords)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Inputs:')
        print(x.eval())

        print('Base indices:')
        print(base_indices.eval())

        print('Offsets:')
        print(offsets.eval())

        print('Coords:')
        print(coords.eval())

        print('Sampled:')
        print(res.eval())


if __name__ == '__main__':
    #_test_simple()
    _test_model()
