#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import logging
import time
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GRUJoint():

    def __init__(self, inputs_val, inputs_len_val, labels_val, config):
        c = config  # shortcut as it is used heavily
        logger.info('Storing data to graph. Inspired by "Preload data" section in https://www.tensorflow.org/versions/r0.7/how_tos/reading_data/index.html')
        logger.info('inputs_val.shape %s, inputs_len_val.shape %s, labels_val.shape %s', inputs_val.shape, inputs_len_val.shape, labels_val.shape)

        start = time.time()
        with tf.Graph().as_default():
            self.dropout_keep_prob = tf.placeholder('float')
            with tf.variable_scope('train_input'):

                turn_initializer = tf.placeholder(shape=inputs_val.shape, dtype=inputs_val.dtype)
                labels_initializer = tf.placeholder(shape=labels_val.shape, dtype=labels_val.dtype)
                turn_len_initialzer = tf.placeholder(shape=inputs_len_val.shape, dtype=inputs_len_val.dtype)

                turn_inputs = tf.Variable(turn_initializer, trainable=False, collections=[])
                turn_lengths = tf.Variable(turn_len_initialzer, trainable=False, collections=[])
                input_labels = tf.Variable(labels_initializer, trainable=False, collections=[])

                turn, turn_len, label = tf.train.slice_input_producer([turn_inputs, turn_lengths, input_labels])
                turns, turn_lens, labels = tf.train.batch([turn, turn_len, label], batch_size=c.batch_size) 

                with tf.variable_scope('graph') as graph_scope:
                    self.tr_logits, self.tr_loss, self.tr_true_count = self.__class__._build_graph(
                            turns, turn_lens, labels, self.dropout_keep_prob, c)

            with tf.variable_scope('dev_input'):
                self.feed_turns = turns =tf.placeholder(shape=[c.batch_size, None], dtype=tf.int64)
                self.feed_labels = labels = tf.placeholder(shape=[c.batch_size, c.labels_size], dtype=tf.int64)
                self.feed_turns_len = turn_lens = tf.placeholder(shape=[c.batch_size, None], dtype=tf.int64)
                # reusing the same parameters from the graph where are trained
                with tf.variable_scope(graph_scope, reuse=True):
                    self.dev_logits, self.dev_loss, self.dev_true_count = self.__class__._build_graph(
                            turns, turn_lens, labels, self.dropout_keep_prob, c)
        logger.info('Loaded time %s', time.time() - start)

    @staticmethod
    def _build_graph(turns, turn_lens, labels, dropout_keep_prob, c):
        logger.debug('turns.get_shape(): %s', turns.get_shape())
        logger.debug('turn_lens.get_shape(): %s', turn_lens.get_shape())
        logger.debug('labels.get_shape(): %s', labels.get_shape())
        with tf.variable_scope('turn_encoder'):
            word_embeddings = tf.get_variable('word_embeddings', 
                    initializer=tf.random_uniform([c.vocab_size, c.embedding_size], -1.0, 1.0))
            embedded_inputs = [tf.nn.embedding_lookup(word_embeddings, tf.squeeze(input_, squeeze_dims=[1])) for input_ in tf.split(1, c.max_seq_len, turns)]
            dropped_embedded_inputs = [tf.nn.dropout(i, dropout_keep_prob) for i in embedded_inputs]

            # FIXME reset RNN state if we see start turn label
            forward_gru = tf.nn.rnn_cell.GRUCell(c.rnn_size, input_size=c.embedding_size)
            logger.debug('c.embedding_size: %s', c.embedding_size)
            logger.debug('dropped_embedded_inputs[0].get_shape(): %s', dropped_embedded_inputs[0].get_shape())

            with tf.variable_scope('forward'):
                outputs, last_state = tf.nn.rnn(
                    cell=forward_gru,
                    inputs=dropped_embedded_inputs,
                    dtype=tf.float32,
                    sequence_length=turn_lens)

        with tf.variable_scope('slot_prediction'):
            # FIXME better initialization
            w_project = tf.get_variable('project2labels', 
                    initializer=tf.random_uniform([c.rnn_size, c.labels_size], -1.0, 1.0))  # FIXME dynamically change size based on the input not used fixed c.rnn_size
            logits = tf.matmul(last_state, w_project)
            logger.debug('last_state.get_shape(): %s', last_state.get_shape())
            logger.debug('w_project.get_shape(): %s', w_project.get_shape())
            logger.debug('logits.get_shape(): %s', logits.get_shape())

        with tf.variable_scope('loss'):
            logger.debug('labels.get_shape(): %s', labels.get_shape())
            logger.debug('tf.to_int64(labels).get_shape(): %s', tf.to_int64(labels).get_shape())
            # xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.to_int64(labels), name='crossentropy')
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='crossentropy')
            loss = tf.reduce_mean(xent, name='xentropy_mean')
            logger.debug('xent.get_shape(): %s', xent.get_shape())
            logger.debug('loss.get_shape(): %s', loss.get_shape())

        with tf.variable_scope('eval'):
            true_count = tf.nn.in_top_k(logits, labels, 1)
            # FIXME compute also accuracy similar like global_step is computed
            logger.debug('true_count.get_shape(): %s', true_count.get_shape())
        return (logits, loss, true_count)
