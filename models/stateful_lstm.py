import numpy as np
import matplotlib.pyplot as plt
import utils
import metrics


def build_model(batch_size, sequence_len, features):
    import keras
    from keras import layers

    model = keras.models.Sequential()
    model.add(layers.LSTM(20, batch_input_shape=(batch_size, sequence_len, features), stateful=False, return_sequences=True))
    #model.add(layers.GRU(20, stateful=True, return_sequences=True))
    #model.add(layers.TimeDistributed(layers.Dense(20)))

    #model.add(keras.layers.Lambda(lambda x: tf.reshape(x, shape=[1, window, 20])))
    model.add(layers.Conv1D(20, 3, padding='same'))
    #model.add(layers.Conv1D(20, 3, padding='same'))

    #model.add(layers.Flatten())
    #model.add(layers.Dense(1, activation='linear'))
    model.add(layers.TimeDistributed(layers.Dense(1)))
    model.compile('adam', 'mse')

    print(model.summary())
    return model


def as_sequences(x: np.ndarray, window_size: int, num_derivatives: int = 0):
    num_sequences = len(x) - window_size
    features = np.zeros((num_sequences, window_size, num_derivatives + 1))
    targets = np.zeros((num_sequences, window_size, 1))
    for i in range(num_sequences):
        # Feature 0 - value of the time series
        features[i, :, 0] = x[i:i + window_size]

        # Feature 1 to 1 + num_derivatives  -  gradients of previous features (inside window)
        for gi in range(num_derivatives):
            features[i, :, gi + 1] = np.gradient(features[i, :, gi])

        targets[i, :, 0] = x[i + 1:i + window_size + 1]
    return features, targets


def sampler(ts: np.ndarray, window=30):
    diff = ts[1:] - ts[:-1]
    x, y = utils.as_sequences(diff, window_size=window)

    for i in range(len(x)):
        yield x[i], y[i]


def main():
    path = 'D:\\ydata-labeled-time-series-anomalies-v1_0\\A4Benchmark\\A4Benchmark-TS3.csv'
    train_test_ratio = 0.6
    batch_size = 64
    epochs = 5
    window = 256

    ts = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]
    ts = (ts - np.mean(ts)) / np.std(ts)

    xaxis = np.linspace(0, len(ts), len(ts))
    ts_train, ts_test = utils.split(ts, ratio=train_test_ratio)
    xaxis_train, xaxis_test = utils.split(xaxis, ratio=train_test_ratio)

    model = build_model(batch_size=batch_size, sequence_len=window, features=1)

    x_train, y_train = as_sequences(ts_train[1:] - ts_train[:-1], window_size=window)
    trunc_len = (len(x_train) // batch_size) * batch_size
    model.fit(x_train[:trunc_len], y_train[:trunc_len], batch_size=batch_size, epochs=epochs)

    x_all, y_all_true = as_sequences(ts[1:] - ts[:-1], window_size=window)
    trunc_len = (len(x_all) // batch_size) * batch_size
    y_all_pred = model.predict(x_all[:trunc_len], batch_size=batch_size)

    plt.plot(y_all_true[:, -1], label='True')
    plt.plot(y_all_pred[:, -1], label='Pred')
    plt.legend()
    plt.grid()
    plt.show()


    """
    # dydt_train = ts_train[1:] - ts_train[:-1]
    # y_train = np.reshape(ts_train[1:], (-1, 1))

    for epoch in range(epochs):
        losses = []
        for _, sample, target in sampler(ts_train, window=window):
            loss = model.train_on_batch(sample, target)
            losses.append(loss)
        print('Epoch {0}/{1}'.format(epoch, epochs))
        print('\tLoss: {0:.3f}'.format(np.mean(losses)))
        print('\tResetting RNN state')
        model.reset_states()
    """

    # prediction / evaluation
    print('Prediction')
    predictions = []

    for original, sample, target in sampler(ts, window=window):
        y = original + model.predict_on_batch(sample)[0]
        predictions.append(y)

    """for i in range(50):
        dydt_pred = model.predict_on_batch(np.reshape([predictions[-1]], (1, 1, 1)))[0] + predictions[-1]
        y = ts[i] + dydt_pred
        predictions.append(y)
        """

    plt.plot(xaxis_train, ts_train, label='Train')
    plt.plot(xaxis_test, ts_test, label='Test')
    plt.plot(np.linspace(window, len(predictions) + window, len(predictions)), predictions, label='Predicted')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
