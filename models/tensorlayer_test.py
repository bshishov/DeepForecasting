import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt


def create_data(size, seq_len):
    seq_len = 64
    ds_size = 100
    sep = ds_size // 2
    x_train = np.repeat(np.expand_dims(np.linspace(0, 12, seq_len), axis=0), repeats=ds_size, axis=0)
    y_train = np.zeros(ds_size)
    x_train[:sep] = np.sin(x_train[:sep])
    y_train[:sep] = 1
    x_train[sep:] = np.cos(x_train[sep:])
    y_train[sep:] = 0
    x_train += np.random.random(x_train.shape) * 0.5
    return x_train, y_train


def main():
    seq_len = 64
    x_train, y_train = create_data(size=100, seq_len=seq_len)

    sess = tf.InteractiveSession()

    # define placeholder
    x = tf.placeholder(tf.float32, shape=(None, ) + x_train.shape[1:], name='x')
    y_ = tf.placeholder(tf.int64, shape=(None, ) + y_train.shape[1:], name='y_')

    net = tl.layers.InputLayer(x, name='input_layer')

    #net = tl.layers.ReshapeLayer(net, (-1, seq_len, 1), name='reshape_rnn')
    #net = tl.layers.RNNLayer(net, cell_fn=tf.contrib.rnn.BasicLSTMCell, n_hidden=32, n_steps=seq_len, return_last=False, name='lstm1')

    # Reshape 1D to 2D for deformable convolutions
    net = tl.layers.ReshapeLayer(net, (-1, seq_len, 1, 1), name='reshape_1d_2d')

    offset = tl.layers.Conv2d(net, 6, (3, 1), (1, 1), act=tf.nn.relu, padding='SAME', name='offset1')
    net = tl.layers.DeformableConv2d(net, offset, n_filter=32, filter_size=(3, 1), act=tf.nn.relu, name='deformable1')

    # Reshape back from 2D to 1D
    net = tl.layers.ReshapeLayer(net, (-1, seq_len, 32), name='reshape_2d_1d')
    net = tl.layers.FlattenLayer(net)
    net = tl.layers.DenseLayer(net, n_units=2, act=tf.identity, name='output_layer')

    # define cost function and metric.
    y = net.outputs
    cost = tl.cost.cross_entropy(y, y_, 'cost')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_op = tf.argmax(tf.nn.softmax(y), 1)

    # define the optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost, var_list=net.all_params)

    # initialize all variables in the session
    tl.layers.initialize_global_variables(sess)

    # print network information
    net.print_params()
    net.print_layers()

    # train the network
    tl.utils.fit(sess, net, train_op, cost, x_train, y_train, x, y_,
                 acc=acc, batch_size=32, n_epoch=10, print_freq=10, eval_train=True)

    offsets = sess.run(offset.outputs, feed_dict={
        x: x_train[:10],
        y_: y_train[:10],
    })
    print(offsets)

    predicted = tl.utils.predict(sess, net, x_train, x, y_op, batch_size=32)
    print(predicted)

    sess.close()


if __name__ == '__main__':
    for x, y in tl.iterate.ptb_iterator(np.arange(100), batch_size=4, num_steps=3):
        print(x)
        print(y)
        print()
    #main()

"""
    

# prepare data
X_train, y_train, X_val, y_val, X_test, y_test = \
                                tl.files.load_mnist_dataset(shape=(-1,784))

# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# define the network
network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
# the softmax is implemented internally in tl.cost.cross_entropy(y, y_, 'cost') to
# speed up computation, so we use identity here.
# see tf.nn.sparse_softmax_cross_entropy_with_logits()
network = tl.layers.DenseLayer(network, n_units=10,
                                act = tf.identity,
                                name='output_layer')
# define cost function and metric.
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# print network information
network.print_params()
network.print_layers()

# train the network
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=500, n_epoch=500, print_freq=5,
            X_val=X_val, y_val=y_val, eval_train=False)

# evaluation
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

# save the network to .npz file
tl.files.save_npz(network.all_params , name='model.npz')
sess.close()

"""
