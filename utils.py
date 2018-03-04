import numpy as np


def split(x: np.ndarray, ratio: float = 0.5, axis=0):
    point = int(float(x.shape[axis] - 1) * ratio)
    return x[:point], x[point:]


def mean_squared_error(a: np.ndarray, b: np.ndarray, axis=None):
    return np.mean(np.square(a - b), axis=axis)


def mean_absolute_error(a: np.ndarray, b: np.ndarray, axis=None):
    return np.mean(np.abs(a - b), axis=axis)


def as_sequences(a: np.ndarray, window: int = 30):
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


class StandardScaler(ScalerBase):
    mean = None
    std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

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
        self.left = x.min(axis=0)
        self.right = x.max(axis=0)

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
        x = np.reshape(a, newshape=(-1, a.shape[-1]))
        self.scaler.fit(x)

    def transform(self, a: np.ndarray) -> np.ndarray:
        x = np.reshape(a, newshape=(-1, a.shape[-1]))
        return np.reshape(self.scaler.transform(x), newshape=a.shape)

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        x = np.reshape(a, newshape=(-1, a.shape[-1]))
        return np.reshape(self.scaler.inverse_transform(x), newshape=a.shape)


def _test_scalers():
    x = np.random.random((100, 20)) * 1000
    for s in [StandardScaler(), MinMaxScaler()]:
        s.fit(x)
        xt = s.transform(x)
        xit = s.inverse_transform(xt)
        print(x - xit < 1e-9)


if __name__ == '__main__':
    _test_scalers()
