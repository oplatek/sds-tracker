#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging
import random

import tensorflow as tf

from tracker.utils import Config
from tracker.fat_model import FatModel

__author__ = 'Petr Belohlavek, Vojtech Hudecek'


if __name__ == '__main__':
    # Don't forget to install bleeding-edge TF version
    # sudo pip3 install http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE\=CPU,TF_BUILD_IS_OPT\=OPT,TF_BUILD_IS_PIP\=PIP,TF_BUILD_PYTHON_VERSION\=PYTHON3,label\=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.wh
    # See Installation for other versions https://github.com/tensorflow/tensorflow/blob/master/README.md

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    input_val = np.array([[1, 2, 2, 4],
                          [1, 3, 3, 4],
                          [1, 0, 0, 4]], dtype='int32')
    # labels_val = np.array([[1, 0, 0],
    #                        [0, 1, 0]], dtype='float32')
    labels_val = np.array([0, 1, 2], dtype='int64')

    assert input_val.shape[0] == labels_val.shape[0]

    c = Config()
    c.seed = 123
    c.epochs = 200
    c.learning_rate = 0.005
    random.seed(c.seed)

    c.vocab_size = np.max(input_val) + 1
    c.batch_size = labels_val.shape[0]
    c.output_dim = np.max(labels_val) + 1
    c.max_seq_len = input_val.shape[1]
    c.embedding_dim = 10
    c.hidden_state_dim = 50
    c.mlp_hidden_layer_dim = 15

    # Fun part
    logger.info('Creating model')
    m = FatModel(c)
    logger.info('Creating optimizer')
    optimizer = tf.train.GradientDescentOptimizer(c.learning_rate)
    logger.info('Creating train_op')
    train_op = optimizer.minimize(m.loss)

    logger.info('Creating session')
    sess = tf.Session()
    logger.info('init')
    init = tf.initialize_all_variables()
    logger.info('Run init')
    sess.run(init)

    train_writer = tf.train.SummaryWriter('log/', sess.graph)

    # Training part
    epoch_loss = 0
    train_summary = None
    for e in range(c.epochs):
        skipped_dial = 0
        for step in range(100):
            _, epoch_loss, train_summary= sess.run([train_op, m.loss, m.tb_info], feed_dict={m.input: input_val, m.labels: labels_val})
            # epoch_loss = loss
        train_writer.add_summary(train_summary, e)

        logger.debug('Average Batch Loss @%d = %f', e, epoch_loss)
        results = sess.run([m.probabilities, m.logits, m.predict, m.accuracy],
                           feed_dict={m.input: input_val, m.labels: labels_val})
        logging.debug('Probs:\n%s', results[0])
        logging.debug('Logits:\n%s', results[1])
        logging.debug('Predict:\n%s', results[2])
        logging.debug('Accuracy:\n%s', results[3])
