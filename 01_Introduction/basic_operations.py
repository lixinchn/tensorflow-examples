'''
Basic Operations example using TensorFlow library.
'''


from __future__ import print_function
import tensorflow as tf


# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.

a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph.
with tf.Session() as sess:
    print('a=2, b=3')
    print('Addition with constants: %i' % sess.run(a + b))
    print('Multiplication with constants: %i' % sess.run(a * b))

# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op.
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a, b)
mul = tf.mul(a, b)

# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print('Addition with variables: %i' % sess.run(add, feed_dict = {a: 2, b: 3}))
    print('Multiplication with variables: %i' % sess.run(mul, feed_dict = {a: 2, b: 3}))


# matrix multiplication
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

