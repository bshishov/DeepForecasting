import numpy as np
import numpy.testing as npt
import unittest
import processing


def assert_arrays_equal(actual, desired):
    npt.assert_almost_equal(actual, desired)


class TestProcessing(unittest.TestCase):
    def test_shift(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        shifted = processing.shift(x, 1, fill_val=np.nan)
        desired = np.array([np.nan, 1, 2, 3, 4])
        assert_arrays_equal(shifted, desired)

        shifted = processing.shift(x, -1, fill_val=np.nan)
        desired = np.array([2, 3, 4, 5, np.nan])
        assert_arrays_equal(shifted, desired)

        shifted = processing.shift(x, -5, fill_val=np.nan)
        desired = np.array([np.nan] * 5)
        assert_arrays_equal(shifted, desired)

        with self.assertRaises(ValueError):
            processing.shift(x, 100)

        with self.assertRaises(ValueError):
            processing.shift(x, -100)

    def test_split(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]

        a, b = processing.split(x, ratio=0.25)
        assert_arrays_equal(a, [1, 2])
        assert_arrays_equal(b, [3, 4, 5, 6, 7, 8])

        a, b = processing.split(x, ratio=0.5)
        assert_arrays_equal(a, [1, 2, 3, 4])
        assert_arrays_equal(b, [5, 6, 7, 8])

        a, b = processing.split(x, ratio=0.75)
        assert_arrays_equal(a, [1, 2, 3, 4, 5, 6])
        assert_arrays_equal(b, [7, 8])

        a, b = processing.split(x, ratio=0.3333333)
        assert_arrays_equal(a, [1, 2])
        assert_arrays_equal(b, [3, 4, 5, 6, 7, 8])

        a, b = processing.split(x, ratio=0)
        assert_arrays_equal(a, [])
        assert_arrays_equal(b, [1, 2, 3, 4, 5, 6, 7, 8])

        a, b = processing.split(x, ratio=1)
        assert_arrays_equal(a, [1, 2, 3, 4, 5, 6, 7, 8])
        assert_arrays_equal(b, [])

        for ratio in np.linspace(0, 1, 100):
            a, b = processing.split(x, ratio=ratio)
            self.assertTrue(len(a) + len(b) == len(x))

        with self.assertRaises(ValueError):
            processing.split(x, ratio=1.1)

        with self.assertRaises(ValueError):
            processing.split(x, ratio=-1.1)

    def split_into_parts(self):
        sip = processing.split_into_parts

        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        desired = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
        ]
        assert_arrays_equal(sip(x, parts=2), desired)
        assert_arrays_equal(sip(x, part_size=5), desired)

        desired = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
        ]
        assert_arrays_equal(sip(x, parts=5), desired)
        assert_arrays_equal(sip(x, part_size=2), desired)

        desired = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ]
        assert_arrays_equal(sip(x, parts=1), desired)
        assert_arrays_equal(sip(x, part_size=10), desired)

        with self.assertRaises(ValueError):
            assert_arrays_equal(sip(x, parts=1, part_size=10), desired)

        x = np.random.random((100, 10, 2, 3))
        s = sip(x, parts=10)
        assert_arrays_equal(s.shape, (10, 10, 2, 3))

    def test_time_difference(self):
        td = processing.time_difference
        assert_arrays_equal([0, 0, 0], td(np.array([1, 1, 1, 1])))
        assert_arrays_equal([0, 0, 0], td(np.array([0, 0, 0, 0])))
        assert_arrays_equal([1, 1, 1], td(np.arange(4)))
        assert_arrays_equal([2, 2, 2], td(np.arange(5), time_lag=2))
        assert_arrays_equal([1, 1, 1, 1], td(np.arange(4), padding='same'))
        assert_arrays_equal([0, 0, 0, 0], td(np.ones(4), padding='same'))
        assert_arrays_equal([-4, -4, 1, 4], td(np.array([4, 0, 1, 5]), padding='same'))

    def test_to_sequences(self):
        ts = processing.to_sequences
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        desired = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9],
        ]
        assert_arrays_equal(desired, ts(x, window_size=3))

        desired = [
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
            [6, 7, 8],
        ]
        assert_arrays_equal(ts(x, window_size=3, offset=2), desired)

        desired = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        assert_arrays_equal(ts(x, window_size=len(x), offset=1), desired)

    def test_from_sequences(self):
        fs = processing.from_sequences
        x = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9],
        ]
        desired = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert_arrays_equal(fs(np.array(x)), desired)

        x = [
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
            [6, 7, 8],
        ]
        desired = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert_arrays_equal(fs(np.array(x), offset=2), desired)

    def test_to_sequences_and_back(self):
        ts = processing.to_sequences
        fs = processing.from_sequences

        x = np.random.random((100, 10))
        assert_arrays_equal(fs(ts(x, window_size=10)), x)
        assert_arrays_equal(fs(ts(x, window_size=1)), x)

    def test_scaler_general(self):
        scaler_classes = [processing.StandardScaler, processing.MinMaxScaler, processing.NoScaler]

        x = np.random.random((100, 10))
        for scaler_cls in scaler_classes:
            scaler = scaler_cls()
            transformed = scaler.fit_transform(x)
            transformed2 = scaler.transform(x)
            assert_arrays_equal(transformed, transformed2)

            inverse_transformed = scaler.inverse_transform(transformed)
            assert_arrays_equal(inverse_transformed, x)

    def test_standard_scaler(self):
        scaler = processing.StandardScaler()
        x = [1, 1, 1, 1, 1]
        transformed = scaler.fit_transform(x)
        assert_arrays_equal(transformed, [0, 0, 0, 0, 0])
        self.assertAlmostEqual(scaler.mean, 1)
        self.assertAlmostEqual(scaler.std, 0)

        scaler = processing.StandardScaler()
        x = [1, 2, 3, 4, 5]
        std = np.sqrt(2)
        transformed = scaler.fit_transform(x)
        assert_arrays_equal(transformed, [-std, -std * 0.5, 0, std * 0.5, std])
        self.assertAlmostEqual(scaler.mean, 3)
        self.assertAlmostEqual(scaler.std, np.sqrt(2))

        scaler = processing.StandardScaler()
        x = np.random.random((100, 10)) * 100 + 100
        transformed = scaler.fit_transform(x)
        self.assertAlmostEqual(transformed.mean(), 0)
        self.assertAlmostEqual(transformed.std(), 1)

    def test_min_max_scaler(self):
        scaler = processing.MinMaxScaler()
        x = [1, 1, 1, 1, 2]
        transformed = scaler.fit_transform(x)
        assert_arrays_equal(transformed, [0, 0, 0, 0, 1])
        self.assertAlmostEqual(scaler.left, 1)
        self.assertAlmostEqual(scaler.right, 2)

        scaler = processing.MinMaxScaler()
        x = [1, 2, 3, 4, 5]
        transformed = scaler.fit_transform(x)
        assert_arrays_equal(transformed, [0, 0.25, 0.5, 0.75, 1])
        self.assertAlmostEqual(scaler.left, 1)
        self.assertAlmostEqual(scaler.right, 5)

        scaler = processing.MinMaxScaler()
        x = np.random.random((100, 10)) * 100 + 100
        transformed = scaler.fit_transform(x)
        self.assertAlmostEqual(transformed.min(), 0)
        self.assertAlmostEqual(transformed.max(), 1)

    def test_no_scaler(self):
        scaler = processing.NoScaler()
        x = np.random.random((100, 10)) * 100 + 100
        transformed = scaler.fit_transform(x)
        assert_arrays_equal(transformed, x)


class TimeSeriesProcessing(unittest.TestCase):
    def test_time_series_inputs(self):
        raw = np.arange(10)
        ts = processing.TimeSeries(raw,
                                   window=2,
                                   predict_steps=1,
                                   predict_gap=1,
                                   return_sequences=False,
                                   train_test_split=0.6,
                                   window_stride=1)
        desired_x = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
        ]
        desired_y = [2, 3, 4, 5, 6, 7, 8, 9]
        assert_arrays_equal(ts.x, desired_x)
        assert_arrays_equal(ts.y, desired_y)
        assert_arrays_equal(ts.t, desired_y)

        ts = processing.TimeSeries(raw, window=2, predict_steps=1,
                                   return_sequences=True, train_test_split=0.6, window_stride=1)
        desired_x = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
        ]
        desired_y = [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9]
        ]
        assert_arrays_equal(ts.x, desired_x)
        assert_arrays_equal(ts.y, desired_y)
        assert_arrays_equal(ts.t, desired_y)

        ts = processing.TimeSeries(raw,
                                   window=2,
                                   predict_steps=3,
                                   return_sequences=False,
                                   train_test_split=0.6,
                                   window_stride=1)
        desired_x = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
        ]
        desired_y = [
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9]
        ]
        assert_arrays_equal(ts.x, desired_x)
        assert_arrays_equal(ts.y, desired_y)
        assert_arrays_equal(ts.t, desired_y)

        ts = processing.TimeSeries(raw,
                                   window=2,
                                   predict_steps=3,
                                   return_sequences=True,
                                   train_test_split=0.6,
                                   window_stride=1)
        desired_x = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
        ]
        desired_y = [
            [[1, 2, 3], [2, 3, 4]],
            [[2, 3, 4], [3, 4, 5]],
            [[3, 4, 5], [4, 5, 6]],
            [[4, 5, 6], [5, 6, 7]],
            [[5, 6, 7], [6, 7, 8]],
            [[6, 7, 8], [7, 8, 9]]
        ]
        assert_arrays_equal(ts.x, desired_x)
        assert_arrays_equal(ts.y, desired_y)
        assert_arrays_equal(ts.t, desired_y)

        ts = processing.TimeSeries(raw,
                                   window=2,
                                   predict_steps=1,
                                   predict_gap=2,
                                   return_sequences=False,
                                   train_test_split=0.6,
                                   window_stride=1)
        desired_x = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
        ]
        desired_y = [3, 4, 5, 6, 7, 8, 9]
        assert_arrays_equal(ts.x, desired_x)
        assert_arrays_equal(ts.y, desired_y)
        assert_arrays_equal(ts.t, desired_y)

        ts = processing.TimeSeries(raw,
                                   window=2,
                                   predict_steps=2,
                                   predict_gap=2,
                                   return_sequences=False,
                                   train_test_split=0.6,
                                   window_stride=1)
        desired_x = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
        ]
        desired_y = [
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9]
        ]
        assert_arrays_equal(ts.x, desired_x)
        assert_arrays_equal(ts.y, desired_y)
        assert_arrays_equal(ts.t, desired_y)

        ts = processing.TimeSeries(raw,
                                   window=2,
                                   predict_steps=2,
                                   predict_gap=2,
                                   return_sequences=True,
                                   train_test_split=0.6,
                                   window_stride=1)
        desired_x = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
        ]
        desired_y = [
            [[2, 3], [3, 4]],
            [[3, 4], [4, 5]],
            [[4, 5], [5, 6]],
            [[5, 6], [6, 7]],
            [[6, 7], [7, 8]],
            [[7, 8], [8, 9]]
        ]
        assert_arrays_equal(ts.x, desired_x)
        assert_arrays_equal(ts.y, desired_y)
        assert_arrays_equal(ts.t, desired_y)


