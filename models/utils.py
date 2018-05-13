import tensorflow as tf
from tensorflow.python.ops import math_ops


def get_loss_by_name(name):
    if isinstance(name, str):
        if 'dodge' in name:
            return huber_dodges_naive_loss
        if 'huber' in name:
            return huber_loss
    return name


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


def huber_dodges_naive_loss(y_true, y_pred, y_naive, alpha=1.0, beta=10.0):
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    y_naive = tf.squeeze(y_naive)
    abs_error = math_ops.abs(y_true - y_pred)
    abs_naive = math_ops.abs(y_true - y_naive)

    # Penalty
    # Computed as sigmoid over relative absolute error to have some area around naive error to avoid.
    # Additional scaling applied over RAE to make avoidance area larger for large naive errors
    # Penalty is scaled by |naive_err| so penalty will have 0 impact when naive are correct predictions
    # Since penalty is sigmoid - the loss will still be monotonic and we don't introduce local optimums
    penalty = alpha * tf.nn.sigmoid(beta * (abs_error - abs_naive) / abs_naive)

    # HUBER  (See losses impl in tf)
    delta = 1.0
    quadratic = math_ops.minimum(abs_error, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear

    return tf.reduce_mean(losses * (1 + penalty))


def naive_from_true(y_true):
    return tf.concat((tf.reduce_mean(y_true, keep_dims=True, axis=-1), y_true[..., :-1]), axis=-1)


def naive_forecast(x: tf.Tensor, return_sequences: bool=True, feature_idx: int=0):
    """
    Naive one-step-ahead "forecasting". Naive means that we're just repeating timeseries at previous timesteps.
    For one-step-ahead it means that we simply pass x as forecasts (or last of x if we're predicting only one point).

    :param x: Input tensor in shape (batch, sequence_len) or (batch, sequence_len, features)
    :param return_sequences: If true - output would be (batch, sequence_len, ...) otherwise - (batch, 1, ...)
    :param feature_idx: If x shape is (batch, sequence_len, features) than feature index is used to get time-series
    :return: Forecasts
    """
    x_shape = x.get_shape()

    # If shape of x is (batch, sequence)
    if len(x_shape) == 2:
        if return_sequences:
            return x
        return x[:, -1]

    # If shape of x is (batch, sequence, features)
    if len(x_shape) == 3:
        if return_sequences:
            return x[:, :, feature_idx]
        return x[:, -1, feature_idx]

    raise ValueError('Incorrect shape for input x')


def _test_naive():
    import numpy as np

    batch = 6
    sequence_len = 10
    x = np.tile(np.arange(sequence_len).reshape(1, sequence_len), (batch, 1))

    x_tf = tf.Variable(x, trainable=False, dtype=tf.float32)
    y_tf_naive_one = naive_forecast(x_tf, return_sequences=False)
    y_tf_naive_seq = naive_forecast(x_tf, return_sequences=True)

    features = 4
    x_f = np.tile(np.arange(sequence_len).reshape(1, sequence_len, 1), (batch, 1, features))
    x_f[:, :, 1:] = -1
    x_tf_f = tf.Variable(x_f, trainable=False, dtype=tf.float32)
    y_tf_f_naive_one = naive_forecast(x_tf_f, return_sequences=False)
    y_tf_f_naive_seq = naive_forecast(x_tf_f, return_sequences=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Without features:')
        print('Naive-seq: {0}'.format(y_tf_naive_seq.eval()))
        print('Naive-one: {0}'.format(y_tf_naive_one.eval()))
        print()
        print('With features:')
        print('Naive-seq: {0}'.format(y_tf_f_naive_seq.eval()))
        print('Naive-one: {0}'.format(y_tf_f_naive_one.eval()))


def _test_losses():
    import numpy as np

    # batch size of 100 and sequence of 10
    y_true = np.expand_dims(np.arange(10, dtype=np.float32), axis=0).repeat(10, axis=0)

    # prediction
    y_pred = y_true + np.random.random(y_true.shape)

    y_true_tf = tf.Variable(y_true, trainable=False, dtype=tf.float32)
    y_pred_tf = tf.Variable(y_pred, trainable=False, dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Naive: {0}'.format(naive_from_true(y_true_tf).eval()))
        print('Huber: {0}'.format(huber_loss(y_true_tf, y_pred).eval()))
        print('Huber dodge 1.0: {0}'.format(huber_dodges_naive_loss(y_true_tf, y_pred_tf).eval()))
        print('Huber dodge 0.0: {0}'.format(huber_dodges_naive_loss(y_true_tf, y_pred_tf, alpha=0.0).eval()))


if __name__ == '__main__':
    _test_naive()
