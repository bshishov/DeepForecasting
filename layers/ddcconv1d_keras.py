import keras
import tensorflow as tf

from layers.ddcconv1d import ddcconv1d, get_shape


class DDCConv1D(keras.layers.Layer):
    """ Dilated Deformable Causal Convolution 1D """
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 offset_mode: str='F',
                 offsets=None,
                 kernel_initializer='glorot_uniform',
                 dilation_rate: int = 1,
                 interpolate: bool=True,
                 **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.offset_mode = offset_mode
        self.kernel = None
        self.offsets = offsets
        self.kernel_initializer = kernel_initializer
        self.dilation_rate = dilation_rate
        self.interpolate = interpolate
        super(DDCConv1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # weights of shape (num_filters, input_channels, kernel_size)
        self.kernel = self.add_weight(name='weights',
                                      shape=(self.filters, input_shape[2], self.kernel_size),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        if self.offsets is None:
            offsets_weights = self.add_weight(name='offsets',
                                           shape=get_shape(self.offset_mode, spec_shape={
                                               'B': input_shape[0],
                                               'S': input_shape[1],
                                               'C': input_shape[2],
                                               'F': self.filters,
                                               'K': self.kernel_size
                                           }),
                                           initializer=keras.initializers.RandomNormal(-2, 0.01),
                                           #initializer=keras.initializers.Zeros(),
                                           trainable=True)
            max_offset = 0.5 * (input_shape[1]) / (self.dilation_rate * self.kernel_size)
            self.offsets = -tf.nn.sigmoid(offsets_weights) * max_offset
        super(DDCConv1D, self).build(input_shape)

    def call(self, x, **kwargs):
        inputs = x
        return ddcconv1d(inputs, self.kernel, self.offsets,
                         interpolate=self.interpolate,
                         offset_mode=self.offset_mode,
                         dilation_rate=self.dilation_rate,
                         name=self.name)

    def compute_output_shape(self, input_shape):
        # Outputs are of shape (batch_size, sequence_len, num_filters)
        return input_shape[0], input_shape[1], self.filters
