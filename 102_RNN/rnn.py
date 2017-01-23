# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


# 下载 mnist 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/', one_hot = True)

# 一张图片是 28 * 28，FNN 是一次性把数据输入到网络，RNN 把它分成块
chunk_size = 28
chunk_n = 28

rnn_size = 256
n_output_layer = 10

X = tf.placeholder('float', [None, chunk_n, chunk_size])
Y = tf.placeholder('float')

# 定义待训练的神经网络
def recurrent_neural_network(data):
	layer = {'w_': tf.Variable(tf.random_normal([rnn_size, n_output_layer])), 'b_': tf.Variable(tf.random_normal([n_output_layer]))}

	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

	data = tf.transpose(data, [1, 0, 2])
	data = tf.reshape(data, [-1, chunk_size])
	data = tf.split(0, chunk_n, data)
	outputs, status = tf.nn.rnn(lstm_cell, data, dtype = tf.float32)

	output = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])
	return output

