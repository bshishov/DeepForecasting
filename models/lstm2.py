import numpy as np
import tensorflow as tf
import tensorlayer as tl
import utils
import processing


class Model(object):
    def __init__(self, batch_size, time_steps, features=1):
        rnn_units = 32

        self.inputs = tf.placeholder(tf.float32, shape=(batch_size, time_steps, features), name='Inputs')
        self.targets = tf.placeholder(tf.float32, shape=(batch_size, time_steps, 1), name='Targets')

        net = tl.layers.InputLayer(self.inputs, name='input_layer')
        self.rnn_layer = tl.layers.RNNLayer(net,
                                            cell_fn=tf.contrib.rnn.BasicLSTMCell,
                                            cell_init_args={'forget_bias': 0.0},
                                            n_hidden=rnn_units,
                                            n_steps=time_steps,
                                            return_last=False,
                                            name='LSTM')
        net = self.rnn_layer
        net = tl.layers.DropoutLayer(net, keep=0.5, name='Dropout', is_fix=True)
        net = tl.layers.Conv1dLayer(net, shape=(3, rnn_units, 64), dilation_rate=1, act=tf.nn.relu, padding='SAME', name='Conv1')
        net = tl.layers.Conv1dLayer(net, shape=(3, 64, 32), dilation_rate=1, act=tf.nn.relu, padding='SAME', name='Conv2')

        net = tl.layers.TimeDistributedLayer(net, tl.layers.DenseLayer, args={
            'n_units': 1,
            'act': tf.identity,
            'name': 'Dense'
        }, name='TDDense')
        self.net = net

        self.predictions = net.outputs

        self.loss = tf.losses.mean_squared_error(self.targets, self.predictions)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        #self.train_op = optimizer.minimize(self.loss, var_list=net.all_params)

        self.sess = tf.InteractiveSession()
        tl.layers.initialize_global_variables(self.sess)

        self.rnn_state = None
        self.reset_rnn_state()

    def reset_rnn_state(self):
        self.rnn_state = tl.layers.initialize_rnn_state(self.rnn_layer.initial_state)

    def train(self, x, y):
        loss, self.rnn_state, _ = self.sess.run([self.loss, self.rnn_layer.final_state, self.train_op], feed_dict={
            self.inputs: x,
            self.targets: y,
            self.rnn_layer.initial_state: self.rnn_state
        })
        return loss

    def predict(self, x):
        predictions = self.sess.run(self.predictions, feed_dict={
            self.inputs: x,
            self.rnn_layer.initial_state: self.rnn_state
        })
        return predictions

    def describe(self):
        #self.net.print_layers()
        self.net.print_params()

    def __del__(self):
        try:
            self.sess.close()
        except:
            pass


def main():
    path = 'D:\\ydata-labeled-time-series-anomalies-v1_0\\A4Benchmark\\A4Benchmark-TS3.csv'
    time_steps = 256
    epochs = 200
    batch_size = 1
    ts = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]

    data = processing.TimeSeries(ts,
                                 window=time_steps,
                                 window_stride=time_steps,
                                 return_sequences=True,
                                 use_time_diff=False,
                                 scaler=utils.StandardScaler)

    model = Model(batch_size=batch_size, time_steps=time_steps, features=1)
    model.describe()

    for epoch in range(epochs):
        losses = []
        for t, x, y in data.train_samples_generator(batch_size=batch_size):
            loss = model.train(x, y)
            losses.append(loss)
        print('Epoch {0}/{1} Loss: {2:.3f}'.format(epoch, epochs, np.mean(losses)))
        model.reset_rnn_state()

    y_true = []
    y_pred = []
    t_all = []
    for t, x_batch, y_batch in data.all_samples_generator(batch_size=batch_size):
        predicted_part = model.predict(x_batch)
        y_pred += list(predicted_part)
        y_true += list(y_batch)
        t_all += list(t)

    y_pred = data.inverse_transform_predictions(np.array(t_all), np.array(y_pred))
    y_true = data.inverse_transform_predictions(np.array(t_all), np.array(y_true))

    import matplotlib.pyplot as plt
    plt.plot(y_pred, label='Predicted')
    plt.plot(y_true, label='True')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
