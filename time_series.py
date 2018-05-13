import numpy as np


def np_pack(rows):
    cols = zip(*rows)
    return [np.array(c) for c in cols]


class TimeSeries(object):
    def __init__(self,
                 data,
                 train_test_split=0.6,
                 test_size=None,
                 scaler=None):
        self.data = data

        self.data_shape = tuple(np.shape(self.data))
        self.dimensions = np.ndim(self.data)

        assert self.dimensions <= 2, 'Number of dimensions of time-series must be 2 or 1'

        self.features = 1 if self.dimensions == 1 else self.data_shape[1]

        # If train size is specified than use it. Otherwise use train_test_split
        self.test_size = test_size if test_size is not None else int(len(self.data) * (1 - train_test_split))
        self.train_size = len(data) - self.test_size

        # Scale data. Scaling is fitted only on train set
        self.scaler = scaler
        if self.scaler is not None:
            self.scaler.fit(self.data[:self.train_size])
            self.data = scaler.transform(self.data)

    def iterate_data(self, start, end, input_window, output_window, offset, stride=None, expand=False):
        stride = stride if stride is not None else output_window
        for i in range(start, end + 1, stride):
            x = self.data[i:i + input_window]
            if expand:
                x = np.expand_dims(x, axis=-1)
            y = self.data[i + offset:i + offset + output_window]
            ty = np.arange(i + offset, i + offset + output_window)
            yield x, y, ty

    def iterate_train(self, input_window, output_window, offset=None, *args, **kwargs):
        offset = offset if offset is not None else input_window
        start = 0
        stop = self.train_size - output_window - offset
        return self.iterate_data(start, stop, input_window, output_window, offset, *args, **kwargs)

    def iterate_test(self, input_window, output_window, offset=None, *args, **kwargs):
        offset = offset if offset is not None else input_window
        start = self.train_size - offset
        stop = len(self.data) - output_window - offset
        return self.iterate_data(start, stop, input_window, output_window, offset, *args, **kwargs)

    def iterate_all(self, input_window, output_window, offset=None, *args, **kwargs):
        offset = offset if offset is not None else input_window
        start = 0
        stop = len(self.data) - output_window - offset
        return self.iterate_data(start, stop, input_window, output_window, offset, *args, **kwargs)

    def train_data(self, *args, **kwargs):
        return np_pack(self.iterate_train(*args, **kwargs))

    def test_data(self, *args, **kwargs):
        return np_pack(self.iterate_test(*args, **kwargs))

    def all_data(self, *args, **kwargs):
        return np_pack(self.iterate_all(*args, **kwargs))

    def get_test_context(self, size, expand=False):
        start = len(self.data) - self.test_size - size
        stop = len(self.data) - self.test_size
        context = self.data[start:stop]
        if expand and self.dimensions == 1:
            return np.expand_dims(context, axis=-1)
        return context

    def get_test_data(self):
        return self.data[-self.test_size:]

    def get_range(self, t):
        pass

    def target_as_input(self, y):
        if np.ndim(y) == 2:
            return np.expand_dims(y, axis=-1)
        return y

    def inverse_y(self, y):
        if self.scaler is not None:
            return self.scaler.inverse_transform(y)
        return y


def predict(context):
    assert len(context) == 10
    return context[-3:] + np.random.random(3) * 2


def predict_batched(context):
    assert context.shape[1] == 10
    return context[:, -3:] + np.random.random(3) * 2


def free_run(predictor, context, n, ts):
    local_ctx = context.copy()
    all_predictions = []
    predicted_points = 0
    while predicted_points < n:
        y_pred = predictor(local_ctx)
        samples = len(y_pred)
        predicted_points += samples
        all_predictions.append(y_pred)
        local_ctx[:-samples] = local_ctx[samples:]
        local_ctx[-samples:] = ts.target_as_input(y_pred)
    return np.concatenate(all_predictions)[:n]


def free_run_batch(predictor_batched, contexts, n, ts, batch_size):
    all_predictions = []
    for ctx_i in range(0, len(contexts), batch_size):
        ctx = contexts[ctx_i:ctx_i + batch_size].copy()
        batch_predictions = []
        predicted_points = 0
        while predicted_points < n:
            y_pred = predictor_batched(ctx)
            samples = y_pred.shape[1]
            predicted_points += samples
            batch_predictions.append(y_pred)
            ctx[:, :-samples] = ctx[:, samples:]
            ctx[:, -samples:] = ts.target_as_input(y_pred)
        all_predictions.append(np.concatenate(batch_predictions, axis=1)[:, :n])
    return np.concatenate(all_predictions, axis=0)


def main():
    import datetime

    start = datetime.datetime.now()
    ts = TimeSeries(np.arange(10000), test_size=10)
    ts.train_data(input_window=10, output_window=3, stride=1)
    end = datetime.datetime.now()

    print(end - start)


    #predictions = free_run_batch(predict_batched, np.random.random((1, 10)), n=10, ts=ts, batch_size=3)
    #print(predictions.shape)
    #print(predictions)


if __name__ == '__main__':
    main()
