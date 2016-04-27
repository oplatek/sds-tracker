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
        self.config = config

        self.input = tf.placeholder('int32', [self.config.batch_size, config.max_seq_len], name='input')
        self.labels = tf.placeholder('int64', [self.config.batch_size], name='labels')
        self.labels_one_hot = tf.one_hot(indices=self.labels,
                                         depth=config.output_dim,
                                         on_value=1.0,
                                         off_value=0.0,
                                         axis=-1)

        self.gru = GRUCell(config.hidden_state_dim)

        embeddings_we = tf.get_variable('word_embeddings', initializer=tf.random_uniform([config.vocab_size, config.embedding_dim], -1.0, 1.0))
        self.emb = embed_input = tf.nn.embedding_lookup(embeddings_we, self.input)
        inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, config.max_seq_len, embed_input)]

        outputs, last_slu_state = tf.nn.rnn(
            cell=self.gru,
            inputs=inputs,
            dtype=tf.float32,)

        w_project = tf.get_variable('project2labels', initializer=tf.random_uniform([config.hidden_state_dim, config.output_dim], -1.0, 1.0))
        self.logits = logits_bo = tf.matmul(last_slu_state, w_project)
        tf.histogram_summary('logits', logits_bo)
        self.probabilities = tf.nn.softmax(logits_bo)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_bo, self.labels_one_hot))
        self.predict = tf.nn.softmax(logits_bo)

        # TensorBoard
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict, 1), self.labels), 'float32'), name='accuracy')
        tf.scalar_summary('CCE loss', self.loss)
        tf.scalar_summary('Accuracy', self.accuracy)
        self.tb_info = tf.merge_all_summaries()
