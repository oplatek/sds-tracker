#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class GRUJoint():
    def __init__(self, config):
        self.input = tf.placeholder("float", [config.batch_size, config.max_seq_len], name='input')
        self.labels = tf.placeholder("float", [config.batch_size, config.labels_size], name='labels')
        self.loss = tf.Variable(6 * np.ones([config.batch_size, 1], dtype=np.float32))
        self.predict = tf.Variable(66 * np.ones([config.batch_size, config.labels_size]))
        self.acc = tf.Variable(666 * np.ones([config.batch_size, 1]))
        self.train_op = tf.Variable(6666 * np.ones([config.batch_size, 666666]))
