#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell

__author__ = 'Petr Belohlavek, Vojtech Hudecek'


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

        embeddings_we = tf.get_variable('word_embeddings', initializer=tf.random_uniform([self.vocab_size, self.emb_dim], -1.0, 1.0))
        self.emb = embed_input = tf.nn.embedding_lookup(embeddings_we, self.input)
        inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, config.max_seq_len, embed_input)]

        outputs, last_slu_state = tf.nn.rnn(
            cell=self.gru,
            inputs=inputs,
            dtype=tf.float32,)

        w_project = tf.get_variable('project2labels', initializer=tf.random_uniform([self.state_dim, self.output_dim], -1.0, 1.0))
        self.logits = logits_bo = tf.matmul(last_slu_state, w_project)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_bo, tf.argmax(self.labels, 1)))
        self.predict = tf.nn.softmax(logits_bo)
