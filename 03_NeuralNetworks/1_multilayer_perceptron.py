from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets('/tmp/data/', one_hot = True)

# parameters
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

# network parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # mnist data input
n_classes = 10 # mnist total classes

# tf graph input
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])

# create model
def multilayer_perceptron(x, weights, biases):
    # hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # hidden layer with RELu activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# construct model
pred = multilayer_perceptron(x, weights, biases)

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# initializing the variables
init = tf.initialize_all_variables()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        # loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # run optimization op(backdrop) and cost op(to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})

            # compute average loss
            avg_cost += c / total_batch

        # display logs per epoch step
        if epoch % display_step == 0:
            print('Epoch: ', '%04d' % (epoch + 1), 'cost= ', '{:.9f}'.format(avg_cost))

    print('optimization finished.')

    # test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
