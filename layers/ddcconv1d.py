import tensorflow as tf
import numpy as np


def get_shape(spec: str, spec_shape: dict):
    return tuple(spec_shape[dim] for dim in spec)


def expand_transform(x, input_spec: str, output_spec: str, output_spec_shape: dict, numpy=False):
    assert len(output_spec) == len(output_spec_shape)

    if numpy:
        tile_op = np.tile
        reshape_op = np.reshape
        transpose_op = np.transpose
    else:
        tile_op = tf.tile
        reshape_op = tf.reshape
        transpose_op = tf.transpose

    input_spec = [dim for dim in input_spec]
    output_spec = [dim for dim in output_spec]
    missing_dims = [dim for dim in output_spec if dim not in input_spec]
    missing_shapes = [output_spec_shape[dim] for dim in missing_dims]
    missing_size = np.prod(missing_shapes)

    tmp_spec = missing_dims + input_spec
    transpose = [tmp_spec.index(dim) for dim in output_spec]

    if len(missing_dims) > 0:
        # Tile and reshape, missing dims will be first
        x = tile_op(x, [missing_size] + [1] * (len(input_spec) - 1))
        x = reshape_op(x, missing_shapes + [output_spec_shape[dim] for dim in input_spec])

    if tmp_spec != output_spec:
        x = transpose_op(x, transpose)
    return x


def get_coords(kernel_indices, offsets, offset_mode: str, spec_shapes: dict):
    """
    Returns float coordinates in (B, S, F, C, K) shape

        kernel_indices in (S, K) shape
        offsets in offset_mode (any combination of dimennsions)
    """
    if offsets is None:
        print('DDCC1D layer used without offsets')
        return expand_transform(kernel_indices, 'SK', 'BSFCK', spec_shapes)

    def _coords(_kernel_indices, _offsets):
        return tf.clip_by_value(_kernel_indices + _offsets, 0, int(_kernel_indices.get_shape()[0]))
    out_spec = 'BSFCK'
    kernel_spec = 'SK'
    if offset_mode == kernel_spec or offset_mode == 'K':
        return expand_transform(_coords(kernel_indices, offsets), kernel_spec, out_spec, spec_shapes)
    if offset_mode == 'S':
        offsets = expand_transform(offsets, offset_mode, kernel_spec, spec_shapes)
        return expand_transform(_coords(kernel_indices, offsets), kernel_spec, out_spec, spec_shapes)
    if offset_mode == 'BSK':
        indices = expand_transform(kernel_indices, kernel_spec, offset_mode, spec_shapes)
        return expand_transform(_coords(indices, offsets), offset_mode, out_spec, spec_shapes)
    if offset_mode == 'SFK':
        indices = expand_transform(kernel_indices, kernel_spec, offset_mode, spec_shapes)
        return expand_transform(_coords(indices, offsets), offset_mode, out_spec, spec_shapes)

    # Naive (most inefficient) method
    indices = expand_transform(kernel_indices, 'SK', out_spec, spec_shapes)
    offsets = expand_transform(offsets, offset_mode, out_spec, spec_shapes)
    return _coords(indices, offsets)


def ddcconv1d(inputs: tf.Variable,
              weights: tf.Variable,
              offsets: tf.Variable,
              dilation_rate: int = 1,
              offset_mode='F',
              interpolate=True,
              name: str='ddcc1d'):
    """
    Deformable Dilated Causal Convolution 1D

    Shape dimensions notation:
        B - batch size
        S - sequence len
        C - input channels
        K - kernel size
        F - filters

    :param name: name of the layer
    :param inputs: Input tensor of shape (B, S, C)
    :param weights: Tensor of shape (F, C, K)
    :param offsets: Tensof of shape (F)
    :param dilation_rate: Size of receptive field gap
    :param offset_mode: offset mode any combination of dimensions, like
        F - one offset per filter,
        FK - offset per filter and each kernel weight
        BSC - offset per every timestep of each sample
    :param interpolate: Use linear interpolation or just convert indices to int32
    :return: Computed 1D convolutions of shape (B, S, F)
    """
    with tf.variable_scope(name):
        batch_size, seq_length, channels = (int(v) for v in inputs.shape)
        filters, _, kernel_size = (int(v) for v in weights.shape)

        spec_shapes = {
            'B': batch_size,
            'S': seq_length,
            'F': filters,
            'C': channels,
            'K': kernel_size
        }

        # Indices stuff
        with tf.variable_scope('KernelBaseIndices'):
            base_indices = np.arange(seq_length).repeat(kernel_size).reshape((-1, kernel_size))
            window_indices = tf.constant(base_indices, dtype=tf.float32, name='window_indices')
            receptive_field = tf.constant(np.linspace(-kernel_size + 1, 0, kernel_size) * dilation_rate,
                                          name='receptive_field',
                                          dtype=tf.float32)
            kernel_indices = window_indices + receptive_field

        with tf.variable_scope('BatchIndices'):
            # Create batch indices constant in BSFCK shape
            batch_indices_np = expand_transform(np.arange(batch_size, dtype=np.int32), 'B', 'BSFCK', spec_shapes, numpy=True)
            batch_indices = tf.constant(batch_indices_np, dtype=tf.int32, name='batch_indices')

        with tf.variable_scope('ChannelIndices'):
            # Create channel indices constant in BSFCK shape
            channel_indices_np = expand_transform(np.arange(channels, dtype=np.int32), 'C', 'BSFCK', spec_shapes, numpy=True)
            channel_indices = tf.constant(channel_indices_np, dtype=tf.int32, name='channel_indices')

        with tf.variable_scope('Sampling'):
            # SAMPLING IS EXTREMELY EXPENSIVE!!!!!
            coords = get_coords(kernel_indices, offsets, offset_mode=offset_mode, spec_shapes=spec_shapes)

            if interpolate:
                # Left and right indices, e.g. index of 3.65 would be 3 on the left and 4 on the right
                indices_left = tf.cast(tf.floor(coords), tf.int32)
                indices_right = tf.cast(tf.ceil(coords), tf.int32)

                # Calculate interpolation, for index 3.65 interpolation factor would be 0.65
                interpolation = coords - tf.cast(indices_left, tf.float32)

                # Sample both values (on the lef and right)
                # Sample input of shape BSC with BSFCK3 indices (produced by stack) -> BSFCK for each side (left and right)
                vals_left = tf.gather_nd(inputs, tf.stack((batch_indices, indices_left, channel_indices), axis=-1))
                vals_right = tf.gather_nd(inputs, tf.stack((batch_indices, indices_right, channel_indices), axis=-1))

                # Interpolated values
                samples = vals_left + (vals_right - vals_left) * interpolation
            else:
                batch_idx = tf.stack((batch_indices, tf.cast(tf.floor(coords), tf.int32), channel_indices), axis=-1)
                samples = tf.gather_nd(inputs, batch_idx)

        with tf.variable_scope('Convolution'):
            # Apply weights: BSFCK * FCK = BSFCK
            conv = samples * weights

            # Sum across kernel: BSFCK -> BSFC
            conv = tf.reduce_sum(conv, axis=-1)

            # Sum across channels: BSFC -> BSF
            conv = tf.reduce_sum(conv, axis=-1)

        return conv


def _transform_test():
    x = np.arange(10)
    spec_shape = {'B': len(x), 'T': 3, 'C': 2}
    outputs = ['BTC', 'BCT', 'TBC', 'TCB', 'CBT', 'CTB']
    for output_spec in outputs:
        print(output_spec)
        print(expand_transform(x, 'B', output_spec, spec_shape, numpy=True))
        print('\n\n')


def _conv_test():
    def print_var(v):
        print("{0}: {1}:\n\t{2}".format(v.name, v.shape, v.eval()))

    # shape: (batch_size, sequence_len, channels)
    x_raw = np.array([
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
    ])

    # Filters: FCK shape
    filter_weights_raw = np.array([
        [[1], [0]]
    ])

    # Filter offsets: F shape
    filter_offsets_raw = np.ones(filter_weights_raw.shape[0]) * 0.5

    x = tf.Variable(x_raw, name='x', dtype=tf.float32, trainable=False)
    filter_weights = tf.Variable(filter_weights_raw, name='filter_weights', dtype=tf.float32)
    filter_offsets = tf.Variable(filter_offsets_raw, name='filter_offsets', dtype=tf.float32)

    y = ddcconv1d(x, weights=filter_weights, offsets=filter_offsets, offset_mode='F')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print_var(x)
        print_var(filter_offsets)
        print_var(filter_weights)
        print_var(y)


def main():
    #_transform_test()
    _conv_test()


if __name__ == '__main__':
    main()
