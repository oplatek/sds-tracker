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

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Config
    c = Config()
    c.seed = 123
    c.epochs = 20
    c.learning_rate = 0.005
    random.seed(c.seed)

    c.vocab_size = 5
    c.labels_size = 3
    c.max_seq_len = 4
    c.batch_size = 1
    c.embedding_dim = 10
    c.hidden_state_dim = 50
    c.mlp_hidden_layer_dim = 10

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

    input_val = np.array([[1, 2, 3, 4],
                          [4, 3, 2, 1]], dtype='int32')
    labels_val = np.array([[1, 0, 0],
                           [0, 1, 0]], dtype='float32')

    # Training part
    for e in range(c.epochs):
        skipped_dial = 0
        for step in range(10000):
            for i in range(input_val.shape[0]):
                _, loss = sess.run([train_op, m.loss], feed_dict={m.input: [input_val[i, :]], m.labels: [labels_val[i, :]]})

            if step % 100 == 0:
                logger.debug('Average Batch Loss @%d/%d = %f', e, step, loss)
                for i in range(input_val.shape[0]):
                    results = sess.run([m.predict, m.logits, m.emb], feed_dict={m.input: [input_val[i, :]]})
                    print(results[2])
                    logging.debug('inputs: \n%s', input_val[i, :])
                    logging.debug('labels: \n%s', labels_val[i, :])
                    logging.debug('Probs:\n%s', results[0])
                    logging.debug('logits:\n%s', results[1])
