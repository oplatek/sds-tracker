#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for dialogue state tracking.

"""
import logging, uuid, random, os
import tensorflow as tf
from tracker.utils import Config, git_info, compare_ref, setup_logging
from tracker.model import GRUJoint
from tracker.training import TrainingOps, EarlyStopper
from tracker.dataset.dstc2 import Dstc2
from tracker.fat_model import FatModel

__author__ = 'Petr Belohlavek, Vojtech Hudecek'


def generateTurn(train_set):
    for i, dialog in enumerate(train_set.dialogs):
        for j, turn in enumerate(dialog):
            # turn[:train_set.turn_lens[i, j]]
            yield turn[:], train_set.labels[i, j]

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    c = Config()  # FIXME load config from json_file after few commints when everybody knows the settings
    c.name='log/%(u)s/%(u)s' % {'u': uuid.uuid1()}
    c.train_dir = c.name + '_traindir'
    c.filename = '%s.json' % c.name
    c.vocab_file = '%s.vocab' % c.name
    c.labels_file = '%s.labels_bd' % c.name
    c.log_name = '%s.log' % c.name
    c.seed=123
    c.train_file = './data/dstc2/data.dstc2.train.json'
    c.dev_file = './data/dstc2/data.dstc2.dev.json'
    c.epochs = 2
    c.sys_usr_delim = ' SYS_USR_DELIM '
    c.learning_rate = 0.00005
    c.validate_every = 200
    c.train_sample_every = 200
    c.batch_size = 1
    c.dev_batch_size = 1
    c.embedding_size=200
    c.dropout=1.0
    c.rnn_size=600
    c.nbest_models=3
    c.not_change_limit = 5  # FIXME Be sure that we compare models from different epochs
    c.sample_unk = 0

    os.makedirs(os.path.dirname(c.name), exist_ok=True)
    setup_logging(c.log_name)
    logger = logging.getLogger(__name__)

    random.seed(c.seed)
    train_set = Dstc2(c.train_file, sample_unk=c.sample_unk, first_n=2 * c.batch_size)
    dev_set = Dstc2(c.dev_file,
            words_vocab=train_set.words_vocab,
            labels_vocab=train_set.labels_vocab,
            max_dial_len=train_set.max_dial_len,
            max_turn_len=train_set.max_turn_len,
            first_n=2 * c.dev_batch_size)


    logger.info('Saving automatically generated stats to config')
    c.git_info = git_info()
    c.max_turn_len = train_set.max_turn_len
    c.max_dial_len = train_set.max_dial_len
    c.vocab_size = len(train_set.words_vocab)
    c.labels_size = len(train_set.labels_vocab)
    logger.info('Config\n\n: %s\n\n', c)
    logger.info('Saving helper files')


    # Don't forget to install bleeding-edge TF version
    # sudo pip3 install http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE\=CPU,TF_BUILD_IS_OPT\=OPT,TF_BUILD_IS_PIP\=PIP,TF_BUILD_PYTHON_VERSION\=PYTHON3,label\=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.wh
    # See Installation for other versions https://github.com/tensorflow/tensorflow/blob/master/README.md

    logger = logging.getLogger(__name__)

    c.log_dir = 'log/'
    c.seed = 123
    c.epochs = 100
    c.learning_rate = 0.005
    random.seed(c.seed)

    c.batch_size = 1
    c.vocab_size = len(train_set.words_vocab)
    c.output_dim = len(train_set._lab_vocab) + 1
    c.max_seq_len = train_set._max_turn_len
    c.max_dialog_len = train_set.max_dial_len
    c.hidden_state_dim = 100
    c.embedding_dim = 300
    c.hidden_state_dim = 250

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

    logger.info('See stats via tensorboard\n\ntensorboard --logdir %s\n\n', c.log_dir)
    train_writer = tf.train.SummaryWriter(c.log_dir, sess.graph)

    # Training part
    epoch_loss = 0
    train_summary = None
    for e in range(c.epochs):
        for dialog, labels in zip(train_set.dialogs, train_set.labels):
            _, epoch_loss, train_summary = sess.run([train_op, m.loss, m.tb_info], feed_dict={m.input_bdt: [dialog],
                                                                                              m.labels_bd: [labels]})
            # epoch_loss = loss
        # for turn, label in generateTurn(train_set):
        #     _, epoch_loss, train_summary = sess.run([train_op, m.loss, m.tb_info], feed_dict={m.input_bdt: [turn], m.labels_bd: [label]})
        #     # epoch_loss = loss
        train_writer.add_summary(train_summary, e)

        logger.debug('Average Batch Loss @%d = %f', e, epoch_loss)
        # results = sess.run([m.probabilities, m.logits, m.predict, m.accuracy],
        #                    feed_dict={m.input_bdt: inp, m.labels_bd: labels_val})
        # logging.debug('Probs:\n%s', results[0])
        # logging.debug('Logits:\n%s', results[1])
        # logging.debug('Predict:\n%s', results[2])
        # logging.debug('Accuracy:\n%s', results[3])


