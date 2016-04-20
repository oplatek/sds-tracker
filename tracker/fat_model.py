#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell

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
        self.gru_n_steps = config.max_seq_len
        self.state_dim = config.hidden_state_dim
        self.output_dim = config.labels_size
        self.vocab_size = config.vocab_size
        self.emb_dim = config.embedding_dim
        self.batch_size = config.batch_size

        self.input = tf.placeholder("int32", [self.batch_size, config.max_seq_len], name='input')
        self.labels = tf.placeholder("float32", [self.batch_size, self.output_dim], name='labels')  # FIXME convert index to one-hot vector in TF

        self.gru = GRUCell(self.state_dim)

        # MLP params
        self.mlp_n_hidden = config.mlp_hidden_layer_dim
        self.mlp_input2hidden_W = tf.Variable(tf.random_normal([self.state_dim, self.mlp_n_hidden]))
        self.mlp_input2hidden_B = tf.Variable(tf.random_normal([self.mlp_n_hidden]))

        self.mlp_hidden2output_W = tf.Variable(tf.random_normal([self.mlp_n_hidden, self.output_dim]))
        self.mlp_hidden2output_B = tf.Variable(tf.random_normal([self.output_dim]))

        # fun part
        embeddings_we = tf.random_uniform([self.vocab_size, self.emb_dim])
        state_bh, logits_o, probability_o = self.one_turn(tf.ones([self.batch_size, self.state_dim]),
                                                          tf.nn.embedding_lookup(embeddings_we, self.input))

        cost = self.one_cost(logits_o, self.labels)

        # boring part
        self.loss = cost
        self.predict = probability_o
        self.logits = logits_o

    def mlp(self, input_layer_bh):
        hidden_layer_bm = tf.nn.sigmoid(tf.matmul(input_layer_bh, self.mlp_input2hidden_W) + broadcast_vector2matrix(self.mlp_input2hidden_B, self.batch_size))
        output_layer_bo = tf.nn.sigmoid(tf.matmul(hidden_layer_bm, self.mlp_hidden2output_W) + broadcast_vector2matrix(self.mlp_hidden2output_B, self.batch_size))
        return output_layer_bo

    def one_turn(self, state_bh, input_bte):
        for step in range(self.gru_n_steps):
            if step > 0:
                tf.get_variable_scope().reuse_variables()
            _output, state_bh = self.gru(input_bte[:,step,:], state_bh)

        logits_bo = self.mlp(state_bh)
        probability_bo = tf.nn.softmax(logits_bo)

        return state_bh, logits_bo, probability_bo

    def one_cost(self, logits_bo, truth_bo):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_bo, truth_bo))
