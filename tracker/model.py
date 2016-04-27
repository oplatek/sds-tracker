#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
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
        self.dropout_keep_prob = tf.placeholder('float')
        with tf.variable_scope('train_input'):

            self.turn_initializer = ti = tf.placeholder(shape=inputs_val.shape, dtype=inputs_val.dtype)
            self.turn_len_initialzer = tli = tf.placeholder(shape=inputs_len_val.shape, dtype=inputs_len_val.dtype)
            self.labels_initializer = li = tf.placeholder(shape=labels_val.shape, dtype=labels_val.dtype)

            self.turn_inputs = tis = tf.Variable(ti, trainable=False, collections=[])
            self.turn_lengths = tls = tf.Variable(tli, trainable=False, collections=[])
            self.input_labels = ils = tf.Variable(li, trainable=False, collections=[])

            dialog, turn_len, label = tf.train.slice_input_producer([tis, tls, ils])
            dialogs, turn_lens, labels = tf.train.batch([dialog, turn_len, label], batch_size=c.batch_size) 

            assert dialogs.get_shape() == (c.batch_size, c.max_dial_len, c.max_turn_len), str(dialog.get_shape(), (c.batch_size, c.max_dial_len, c.max_turn_len))
            assert turn_lens.get_shape() == [c.batch_size, c.max_dial_len], str(turn_lens.get_shape(), [c.batch_size, c.max_dial_len])
            assert labels.get_shape() == [c.batch_size, c.max_dial_len], str(labels.get_shape() == [c.batch_size, c.max_dial_len])

            with tf.variable_scope('graph') as graph_scope:
                self.tr_predictss, self.tr_loss, self.tr_dial_len, self.tr_true_count, batch_acc = self.__class__._build_graph(
                        dialogs, turn_lens, labels, self.dropout_keep_prob, c)
                # tf.scalar_summary('train/loss', self.tr_loss)
                # tf.scalar_summary('train/batch_accuracy', batch_acc)
                # tf.scalar_summary('train/true_count', self.tr_true_count)
                # tf.scalar_summary('train/dial_len', self.tr_dial_len)

        with tf.variable_scope('dev_input'):
            self.feed_dials = dialogs =tf.placeholder(shape=[c.dev_batch_size, c.max_dial_len, c.max_turn_len], dtype=tf.int64)
            self.feed_labels = labels = tf.placeholder(shape=[c.dev_batch_size, c.max_dial_len], dtype=tf.int64)
            self.feed_turn_lens = turn_lens = tf.placeholder(shape=[c.dev_batch_size, c.max_dial_len], dtype=tf.int64)
            # reusing the same parameters from the graph where are trained
            with tf.variable_scope(graph_scope, reuse=True):
                self.dev_predictss, self.dev_loss, self.dev_dial_len, self.dev_true_count, batch_acc = self.__class__._build_graph(
                        dialogs, turn_lens, labels, self.dropout_keep_prob, c)
                # tf.scalar_summary('dev/loss', self.dev_loss)
                # tf.scalar_summary('dev/batch_accuracy', batch_acc)
                # tf.scalar_summary('dev/true_count', self.dev_true_count)
                # tf.scalar_summary('dev/dial_len', self.dev_dial_len)
        logger.info('Loaded time %s', time.time() - start)

    @staticmethod
    def _build_graph(dialog, turn_lens, labels, dropout_keep_prob, c):
        slu_states = [666] * c.max_dial_len
        for t in range(c.max_dial_len):
            # FIXME separate into function
            reuse_it = True if t > 0 else None
            with tf.variable_scope('turn_encoder', reuse=reuse_it):
                word_embeddings = tf.get_variable('word_embeddings', 
                        initializer=tf.random_uniform([c.vocab_size, c.embedding_size], -1.0, 1.0))
                embedded_inputs = [tf.nn.embedding_lookup(word_embeddings, tf.squeeze(input_, squeeze_dims=[1])) for input_ in tf.split(1, c.max_turn_len, dialog[:, t, :])]
                dropped_embedded_inputs = [tf.nn.dropout(i, dropout_keep_prob) for i in embedded_inputs]

                forward_slu_gru = tf.nn.rnn_cell.GRUCell(c.rnn_size, input_size=c.embedding_size)
                logger.debug('c.embedding_size: %s', c.embedding_size)
                logger.debug('dropped_embedded_inputs[0].get_shape(): %s', dropped_embedded_inputs[0].get_shape())
                with tf.variable_scope('forward_slu'):
                    outputs, last_slu_state = tf.nn.rnn(
                        cell=forward_slu_gru,
                        inputs=dropped_embedded_inputs,
                        dtype=tf.float32,
                        sequence_length=turn_lens[:, t])
                    slu_states[t] = last_slu_state

        masked_turns = tf.to_int64(tf.greater(turn_lens, tf.zeros_like(turn_lens)))
        logger.debug('masked_turns.get_shape(): %s', masked_turns.get_shape())
        dial_len = tf.reduce_sum(masked_turns, 1)
        masked_turnsf = tf.to_float(masked_turns)

        forward_dst_gru = tf.nn.rnn_cell.GRUCell(c.rnn_size, input_size=c.rnn_size)  # FIXME use different rnn_size
        with tf.variable_scope('dialog_state'):
                dialog_states, last_dial_state = tf.nn.rnn(
                    cell=forward_dst_gru,
                    inputs=slu_states,
                    dtype=tf.float32,
                    sequence_length=dial_len)
        logitss = [444] * c.max_dial_len
        for t in range(c.max_dial_len):
            with tf.variable_scope('slot_prediction', reuse=True if t > 0 else None):
                # FIXME better initialization
                w_project = tf.get_variable('project2labels', 
                        initializer=tf.random_uniform([c.rnn_size, c.labels_size], -1.0, 1.0))  # FIXME dynamically change size based on the input not used fixed c.rnn_size
                logitss[t] = tf.matmul(dialog_states[t], w_project)

        logger.debug('dialog_states[0].get_shape(): %s', dialog_states[0].get_shape())
        logger.debug('w_project.get_shape(): %s', w_project.get_shape())

        logits = tf.reshape(tf.concat(1, logitss), (np.prod(masked_turns.get_shape().as_list()), c.labels_size)) 
        logger.debug('logits.get_shape(): %s', logits.get_shape())

        with tf.variable_scope('loss'):
            logger.debug('labels.get_shape(): %s', labels.get_shape())
            masked_logits = tf.mul(logits, tf.reshape(masked_turnsf, (np.prod(masked_turnsf.get_shape().as_list()), 1)))
            logger.debug('masked_logits.get_shape(): %s, masked_logits.dtype %s', masked_logits.get_shape(), masked_logits.dtype)
            labels_vec = tf.reshape(labels, [-1])
            xents = tf.nn.sparse_softmax_cross_entropy_with_logits(masked_logits, labels_vec)
            logger.debug('xents.get_shape(): %s, dtype %s', xents.get_shape(), xents.dtype)
            loss = tf.reduce_sum(xents) / tf.reduce_sum(masked_turnsf)

        with tf.variable_scope('eval'):
            predicts = tf.argmax(masked_logits, 1)
            true_count = tf.reduce_sum(tf.to_int64(tf.equal(predicts, labels_vec)) * tf.reshape(masked_turns, [-1]))
            batch_accuracy = tf.div(true_count, tf.reduce_sum(dial_len))
            logger.debug('true_count.get_shape(): %s', true_count.get_shape())
        logger.info('trainable variables: %s', '\n'.join([str((v.name, v.get_shape())) for v in tf.trainable_variables()]))
        return (predicts, loss, dial_len, true_count, batch_accuracy)
