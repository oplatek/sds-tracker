#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.functional_ops import scan


__author__ = 'Petr Belohlavek, Vojtech Hudecek'


def broadcast_vector2matrix(vector, batch_size):
    return tf.tile(tf.expand_dims(vector, 0), [batch_size, 1])


class FatModel:
    """
    b = batch size
    h = hidden state dimension
    e = embedding dimension
    t = input length
    o = one-hot target variable dimension
    m = MLP hidden layer dimension
    """
    def __init__(self, config):
        self.config = config

        self.input = tf.placeholder('int32', [self.config.batch_size, config.max_seq_len], name='input')
        self.labels = tf.placeholder('int64', [self.config.batch_size], name='labels')
        self.labels_one_hot = tf.one_hot(indices=self.labels,
                                         depth=config.output_dim,
                                         on_value=1.0,
                                         off_value=0.0,
                                         axis=-1)

        self.gru = GRUCell(self.config.hidden_state_dim)

        # MLP params
        self.mlp_input2hidden_W = tf.Variable(tf.random_normal([self.config.hidden_state_dim,
                                                                self.config.mlp_hidden_layer_dim]))
        self.mlp_input2hidden_B = tf.Variable(tf.random_normal([self.config.mlp_hidden_layer_dim]))

        self.mlp_hidden2output_W = tf.Variable(tf.random_normal([self.config.mlp_hidden_layer_dim,
                                                                 self.config.output_dim]))
        self.mlp_hidden2output_B = tf.Variable(tf.random_normal([self.config.output_dim]))

        # fun part
        self.embeddings_we = tf.random_uniform([self.config.vocab_size, self.config.embedding_dim])
        state_bh, logits_bo, probability_bo = self.one_turn(tf.ones([self.config.batch_size,
                                                                     self.config.hidden_state_dim]),
                                                            tf.nn.embedding_lookup(self.embeddings_we, self.input))

        self.loss = self.one_cost(logits_bo, self.labels_one_hot)
        self.probabilities = probability_bo
        self.predict = tf.argmax(probability_bo, 0, name='prediction')
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.labels), 'float32'), name='accuracy')
        self.logits = logits_bo

        # TensorBoard
        tf.scalar_summary('CCE loss', self.loss)
        tf.scalar_summary('Accuracy', self.accuracy)
        self.tb_info = tf.merge_all_summaries()

    def mlp(self, input_layer_bh):
        hidden_layer_bm = tf.nn.sigmoid(tf.matmul(input_layer_bh, self.mlp_input2hidden_W) +
                                        broadcast_vector2matrix(self.mlp_input2hidden_B, self.config.batch_size))

        tf.histogram_summary('MLP Hidden', hidden_layer_bm)

        output_layer_bo = tf.nn.sigmoid(tf.matmul(hidden_layer_bm, self.mlp_hidden2output_W) +
                                        broadcast_vector2matrix(self.mlp_hidden2output_B, self.config.batch_size))
        tf.histogram_summary('MLP Output', output_layer_bo)

        return output_layer_bo

    def one_turn(self, state_bh, input_bte):
        state_tbh = scan(fn=lambda last_state_bh, curr_input_bte: self.gru(curr_input_bte, last_state_bh)[1],
                         elems=tf.transpose(input_bte, perm=[1, 0, 2]),
                         initializer=state_bh)
        state_bh = state_tbh[state_tbh.get_shape()[0]-1, :, :]

        logits_bo = self.mlp(state_bh)
        probability_bo = tf.nn.softmax(logits_bo)

        return state_bh, logits_bo, probability_bo

    def one_cost(self, logits_bo, truth_bo):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_bo, truth_bo))
