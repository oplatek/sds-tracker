#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.functional_ops import scan
__author__ = 'Petr Belohlavek, Vojtech Hudecek'


class FatModel:
    """
    b = batch size
    d = dialog size (number of turns)
    h = hidden state dimension
    e = embedding dimension
    t = input_bdt length (number of tokens)
    o = one-hot target variable dimension
    m = MLP hidden layer dimension
    w = volabulary size
    """
    def __init__(self, config):
        self.config = config

        self.input_bdt = tf.placeholder('int32', [self.config.batch_size,
                                                  self.config.max_dialog_len,
                                                  config.max_seq_len],
                                        name='input_bdt')
        self.labels_bd = tf.placeholder('int64', [self.config.batch_size, self.config.max_dialog_len], name='labels_bd')
        self.onehot_labels_bdo = tf.one_hot(indices=self.labels_bd,
                                            depth=config.output_dim,
                                            on_value=1.0,
                                            off_value=0.0,
                                            axis=-1)

        self.gru = GRUCell(config.hidden_state_dim)

        embeddings_we = tf.get_variable('word_embeddings', initializer=tf.random_uniform([config.vocab_size, config.embedding_dim], -1.0, 1.0))
        embedded_input_bdte = tf.nn.embedding_lookup(embeddings_we, self.input_bdt)
        # inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, config.max_seq_len, embedded_input_bdte)]

        self.projection_ho = tf.get_variable('project2labels', initializer=tf.random_uniform([config.hidden_state_dim, config.output_dim], -1.0, 1.0))

        state_dbh = scan(fn=self.one_turn,
                         elems=tf.transpose(embedded_input_bdte, perm=[1, 0, 2, 3]),
                         initializer=tf.ones([self.config.batch_size, self.config.hidden_state_dim]))

        state_bh = state_dbh[state_dbh.get_shape()[0] - 1, :, :]

        self.logits = logits_bo = tf.matmul(state_bh, self.projection_ho)
        self.probabilities = tf.nn.softmax(logits_bo)
        final_onehot_bo = self.onehot_labels_bdo[:, self.onehot_labels_bdo.get_shape()[1]-1, :]
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_bo, final_onehot_bo))
        self.predict = tf.nn.softmax(logits_bo)

        # TensorBoard
        # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict, 1), self.labels_bd), 'float32'), name='accuracy')
        tf.scalar_summary('CCE loss', self.loss)
        # tf.scalar_summary('Accuracy', self.accuracy)
        tf.histogram_summary('logits', logits_bo)
        self.tb_info = tf.merge_all_summaries()

    def one_turn(self, last_state_bh, turn_input_bte):
        state_tbh = scan(fn=lambda last_state_bh, curr_input_bte: self.gru(curr_input_bte, last_state_bh)[1],
                         elems=tf.transpose(turn_input_bte, perm=[1, 0, 2]),
                         initializer=last_state_bh)

        state_bh = state_tbh[state_tbh.get_shape()[0] - 1, :, :]

        logits_bo = tf.matmul(state_bh, self.projection_ho)
        probs_bo = tf.nn.softmax(logits_bo)

        return state_bh #, logits_bo, probs_bo

        # outputs, next_state = tf.nn.rnn(cell=self.gru,
        #                                 inputs=turn_input_bte,
        #                                 initial_state=last_state_bh,
        #                                 dtype=tf.float32)
        # return next_state
