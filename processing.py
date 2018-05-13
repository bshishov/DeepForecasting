import numpy as np
import random


def shift(x: np.ndarray, offset: int, fill_val=np.nan):
    if abs(offset) > len(x):
        raise ValueError('Offset should be less or equal than length of x')
    if offset >= 0:
        return np.concatenate((np.full(offset, fill_val, dtype=x.dtype), x[:-offset]))
    else:
        return np.concatenate((x[-offset:], np.full(-offset, fill_val, dtype=x.dtype)))


def split(x: list, ratio: float = 0.5) -> (np.ndarray, np.ndarray):
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError('Ratio should be in range [0, 1]')
    point = int(len(x) * ratio)
    return x[:point], x[point:]


def as_sequences_fast(a: np.ndarray, window: int = 30):
    """
    Converts an array of shape (n, ...) to (n - window, window, ...) by performing sliding window
    Done by using stride tricks in NumPy so it is done internally with O(1) complexity.

    NOTE: first axis should be samples

    :param a: Original array of shape (n, ...)
    :param window: Window size, e.g. sequence length
    :return: Sequenced array (n - window, window, ...)
    """
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0], ) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


class ScalerBase(object):
    def fit(self, x: np.ndarray):
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, x: np.ndarray)-> np.ndarray:
        self.fit(x)
        return self.transform(x)


class StandardScaler(ScalerBase):
    mean = None
    std = None

    def fit(self, x: np.ndarray):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None
        assert self.std is not None

        return (x - self.mean) / (self.std + 1e-10)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None
        assert self.std is not None

        return x * self.std + self.mean


class MinMaxScaler(ScalerBase):
    left = None
    right = None

    def fit(self, x: np.ndarray):
        self.left = np.min(x, axis=0)
        self.right = np.max(x, axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.left is not None
        assert self.right is not None

        return (x - self.left) / (self.right - self.left)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self.left is not None
        assert self.right is not None

        return x * (self.right - self.left) + self.left


class NoScaler(ScalerBase):
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def fit(self, x: np.ndarray):
        pass


class SequenceScaler(ScalerBase):
    def __init__(self, scaler_cls: type = StandardScaler):
        self.scaler = scaler_cls()

    def fit(self, a: np.ndarray):
        # a - has shape (N, window_size, features)

        # Take only last element in window
        # b = a[:, -1, :]
        b = a

        # Reshape to (N, features)
        x = np.reshape(b, newshape=(-1, b.shape[-1]))
        self.scaler.fit(x)

    def transform(self, a: np.ndarray) -> np.ndarray:
        x = np.reshape(a, newshape=(-1, a.shape[-1]))
        return np.reshape(self.scaler.transform(x), newshape=a.shape)

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        x = np.reshape(a, newshape=(-1, a.shape[-1]))
        return np.reshape(self.scaler.inverse_transform_predictions(x), newshape=a.shape)


def roll_cv(x: np.ndarray,
            y: np.ndarray=None,
            folds: int = 4,
            backtrack_padding: int = 0,
            max_train_size=None):
    """
    Splits the data into parts sequentially. Useful for time-series validation.

    For example, given array [0, 1, 2, 3, 4, 6] and folds 2:
        The fold size will be 2
        Fold 0 returns: [0, 1], [2, 3]
        Fold 1 returns: [0, 1, 2, 3], [4, 5]

    :param x: Input array
    :param y: Optional second array to split, if not none, function will return x_train, x_test, y_train, y_test
    :param folds: Number of total folds to create
    :param backtrack_padding: Number of points to prepend to each return, note that fold size will be changed
    :param max_train_size: Limits the training part by some size, if None no limitation performed
    :return: Yields training and testing parts @folds times.
    """
    assert folds >= 1
    if y is not None:
        assert len(x) == len(y)
    fold_len = (len(x) - backtrack_padding) // (folds + 1)
    for fold in range(folds):
        sep = backtrack_padding + (fold + 1) * fold_len
        if max_train_size is not None:
            start = max(0, sep - max_train_size)
        else:
            start = 0
        if y is not None:
            # X Train, X test, y train, y test
            yield x[start:sep], x[sep - backtrack_padding:sep + fold_len], \
                  y[start:sep], y[sep - backtrack_padding:sep + fold_len]
        else:
            # X Train, X test
            yield x[start:sep], x[sep - backtrack_padding:sep + fold_len]


def as_sequences(x: np.ndarray, window_size: int, num_derivatives: int = 0, points_to_predict: int = 1):
    """
    Transforms 1d float time-series array x of length N
    to:
        array of input features:  (N - window_size + 1 - points_to_predict, window_size, 1 + num_derivatives)
        array of expected output targets:  (N - window_size + 1 - points_to_predict, points_to_predict)
    :param x: input time-series
    :param window_size: size of the sliding window
    :param num_derivatives: number of derivatives to compute inside window
    :param points_to_predict: number of target point per sample
    :return: features, targets
    """
    num_sequences = len(x) - window_size + 1 - points_to_predict
    features = np.zeros((num_sequences, window_size, num_derivatives + 1))
    targets = np.zeros((num_sequences, points_to_predict))
    for i in range(num_sequences):
        # Feature 0 - value of the time series
        features[i, :, 0] = x[i:i + window_size]

        # Feature 1 to 1 + num_derivatives  -  gradients of previous features (inside window)
        for gi in range(num_derivatives):
            features[i, :, gi + 1] = np.gradient(features[i, :, gi])

        # Targets values (n points)
        # for pi in range(points_to_predict):
            # targets[i, pi] = x[i + window_size + pi]

        targets[i, :] = x[i + window_size:i + window_size + points_to_predict]
    return features, targets


def scaled_grads(a: np.ndarray, axis: int = 0):
    grads = np.gradient(a, axis=axis)
    return (grads - grads.mean()) / (grads.std() + 1e-10)


def split_into_parts(a: np.ndarray, parts: int=None, part_size: int=None):
    if parts is None and part_size is None:
        raise ValueError('You should specify only parts or part_size')
    if parts is not None and part_size is not None:
        raise ValueError('You should specify only parts or part_size')

    if parts is not None:
        part_size = len(a) // parts

    if part_size is not None:
        parts = len(a) // part_size

    assert parts > 1
    assert part_size > 0

    res = np.empty((parts, part_size) + a.shape[1:], dtype=a.dtype)

    for i in range(parts):
        res[i, :] = a[i * part_size:(i + 1) * part_size]
    return res


def time_difference(a: np.ndarray, time_lag: int=1, padding='valid'):
    if padding == 'valid':
        return a[time_lag:] - a[:-time_lag]

    if padding == 'same':
        assert time_lag == 1
        return np.concatenate(([a[1] - a[0], ], a[1:] - a[:-1]))

    raise ValueError('Padding should be \'valid\' or \'same\', got: {0}'.format(padding))


def derivatives_deepcast(x: np.ndarray, num_derivatives: int = 2):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=-1)

    samples, sequence_len, features = x.shape
    assert features == 1
    x = np.repeat(x, num_derivatives + 1, axis=-1)
    for i in range(num_derivatives):
        x[:, :-1, i + 1] = x[:, 1:, i] - x[:, :-1, i]
        x[:, -1, i + 1] = np.mean(x[:, :-1, i + 1])
    return x


def to_sequences(array: np.ndarray, window_size: int, offset: int = 1):
    assert 1 <= offset <= len(array) and 1 <= window_size <= len(array)
    num_sequences = (len(array) - window_size + offset) // offset
    sequences = np.empty((num_sequences, window_size) + array.shape[1:], dtype=array.dtype)
    for i in range(num_sequences):
        sequences[i] = array[i * offset:i * offset + window_size]
    return sequences


def from_sequences(array: np.ndarray, offset: int = 1):
    assert offset >= 1 and len(array.shape) >= 2
    num_sequences, window_size = array.shape[0:2]
    output_len = offset * num_sequences + window_size - offset
    output = np.empty((output_len, ) + array.shape[2:], dtype=array.dtype)
    output[:window_size] = array[0]
    output[window_size:] = array[1:, -offset:].reshape((-1, ) + array.shape[2:])
    return output


def batch_generator(*data, batch_size: int = 32, drop_last: bool = True, shuffle: bool = False):
    total_len = min([len(seq) for seq in data])
    indices = list(range(total_len))

    if shuffle:
        random.shuffle(indices)

    for i in range(0, total_len, batch_size):
        if drop_last and i + batch_size > total_len:
            break
        batch_indices = indices[i:i + batch_size]
        yield (seq[batch_indices] for seq in data)


class TimeSeries(object):
    def __init__(self,
                 time_series: np.ndarray,
                 window: int = 256,
                 window_stride: int = 1,
                 predict_gap: int = 1,
                 predict_steps: int = 1,
                 num_derivatives: int = 0,
                 return_sequences: bool = False,
                 deepcast_derivatives: bool = False,
                 train_test_split: float = 0.6,
                 scaler=None,
                 use_time_diff: bool = False):
        self.series_raw = time_series
        self.window = window
        self.train_test_split = train_test_split
        self.use_time_diff = use_time_diff
        self.return_sequences = return_sequences
        self.window_stride = window_stride
        self.predict_steps = predict_steps
        self.predict_gap = predict_gap
        self.scaler_x = scaler
        if scaler is None:
            self.scaler_y = None
        else:
            self.scaler_y = type(scaler)()
        time = np.arange(len(self.series_raw), dtype=np.uint16)

        if use_time_diff:
            series = time_difference(self.series_raw)
            time = time[1:]
        else:
            series = self.series_raw

        time = time[self.predict_gap:]
        target = series[self.predict_gap:]

        if not deepcast_derivatives and num_derivatives > 0:
            if series.ndim == 1:
                series = np.expand_dims(series, axis=-1)

            assert series.shape[1] == 1

            series = np.repeat(series, num_derivatives + 1, axis=-1)
            for i in range(num_derivatives):
                series[:, i + 1] = time_difference(series[:, i], padding='same')

        if self.predict_steps > 1:
            time = to_sequences(time, window_size=self.predict_steps, offset=1)
            target = to_sequences(target, window_size=self.predict_steps, offset=1)

        series = series[:-self.predict_gap - self.predict_steps + 1]

        assert len(series) == len(target) == len(time)

        if self.scaler_x is not None:
            series_train, _ = split(series, ratio=train_test_split)
            self.scaler_x.fit(series_train)
            series = self.scaler_x.transform(series)

        if self.scaler_y is not None:
            target_train, _ = split(target, ratio=train_test_split)
            self.scaler_y.fit_transform(target_train)
            target = self.scaler_y.transform(target)

        self.x = to_sequences(series, window_size=self.window, offset=self.window_stride)

        if deepcast_derivatives and num_derivatives > 0:
            self.x = derivatives_deepcast(self.x, num_derivatives=num_derivatives)

        if return_sequences:
            self.t = to_sequences(time, window_size=self.window, offset=self.window_stride)
            self.y = to_sequences(target, window_size=self.window, offset=self.window_stride)
        else:
            self.t = time[-len(self.x):]
            self.y = target[-len(self.x):]

        assert len(self.x) == len(self.y) == len(self.t)

        self.x_train, self.x_test = split(self.x, ratio=train_test_split)
        self.y_train, self.y_test = split(self.y, ratio=train_test_split)
        self.t_train, self.t_test = split(self.t, ratio=train_test_split)

    @property
    def input_shape(self):
        return self.x.shape[1:]

    @property
    def output_shape(self):
        return self.y.shape[1:]

    def train_samples_generator(self, *args, **kwargs):
        return batch_generator(self.t_train, self.x_train, self.y_train, *args, **kwargs)

    def test_samples_generator(self, *args, **kwargs):
        return batch_generator(self.t_test, self.x_test, self.y_test, *args, **kwargs)

    def all_samples_generator(self, *args, **kwargs):
        return batch_generator(self.t, self.x, self.y, *args, **kwargs)

    def inverse_transform_predictions(self, time: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        assert time.shape[0] == predictions.shape[0]

        if self.return_sequences:
            time = from_sequences(time, self.window_stride)
            predictions = from_sequences(predictions, self.window_stride)

        """
        _t = time.flatten()
        _x = predictions.flatten()
        unique_t, indices = np.unique(_t, return_index=True)

        y = np.zeros(len(unique_t), dtype=predictions.dtype)
        y[indices] = _x[indices]
        """

        y = self.inverse_target_scale(predictions)

        if self.use_time_diff:
            return y + self.series_raw[time - 2]
        return y

    def inverse_target_scale(self, targets: np.ndarray) -> np.ndarray:
        if self.scaler_y is not None:
            return self.scaler_y.inverse_transform(targets)
        return targets


def _test_to_sequences():
    seq = to_sequences(np.arange(100, dtype=np.uint8), window_size=20, offset=1)
    print(seq)

    original = from_sequences(seq, offset=1)
    print(original)


def _test_time_series():
    raw_ts = np.sin(np.linspace(0, 6.3, 100))
    data = TimeSeries(raw_ts,
                      window=10,
                      window_stride=10,
                      return_sequences=False,
                      use_time_diff=True,
                      scaler=StandardScaler)
    print(data.input_shape)
    print(data.output_shape)
    print('\n\n\n')
    print(data.x[0])
    print(data.y[0])

    print('\n\n\n')
    print(data.x[1])
    print(data.y[1])

    print('\n\n\n')
    print(data.series_raw[:12])

    import matplotlib.pyplot as plt
    batch = 0
    for t, x, y in data.all_samples_generator(batch_size=1, shuffle=False):
        _t = t.flatten()
        _x = x.flatten()
        indices = sorted(list(range(len(_x))), key=lambda i: _t[i])
        plt.plot(_t[indices], _x[indices], label='Flat Batch {0}'.format(batch))
        batch += 1
    plt.legend()
    plt.show()


def _test_as_sequences():
    x, y = as_sequences(np.arange(10), window_size=3, points_to_predict=3, num_derivatives=2)
    print(x)
    print(x.shape)
    print()
    print(y)
    print(y.shape)


def _test_roll_cv():
    a = np.arange(6)
    for x_train, x_test, y_train, y_test in roll_cv(a, a, folds=2):
        print('FOLD')
        print('X TRAIN', x_train)
        print('X TEST', x_test)
        print('Y TRAIN', y_train)
        print('Y TEST', y_test)


def _test_scalers():
    x = np.random.random((100, 20)) * 1000
    for s in [StandardScaler(), MinMaxScaler()]:
        s.fit(x)
        xt = s.transform(x)
        xit = s.inverse_transform(xt)
        print(x - xit < 1e-9)


if __name__ == '__main__':
    _test_to_sequences()
    #_test_as_sequences()
    #_test_scalers()
    #_test_roll_cv()
    # _test_as_sequences()
