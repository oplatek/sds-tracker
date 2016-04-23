#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging, uuid, random
import tensorflow as tf
import numpy as np
from tracker.utils import Config, git_info
from tracker.model import GRUJoint
from tracker.training import TrainingOps, EarlyStopper
from tracker.dataset.dstc2 import Dstc2


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

c = Config()  # FIXME load config from json_file after few commints when everybody knows the settings
c.name='log/demo_%s' % uuid.uuid1()
c.train_dir = c.name + '_traindir'
c.filename = '%s.json' % c.name
c.vocab_file = '%s.vocab' % c.name
c.labels_file = '%s.labels' % c.name
c.seed=123
c.train_file = './data/dstc2/data.dstc2.train.json'
c.dev_file = './data/dstc2/data.dstc2.dev.json'
c.epochs = 2
c.sys_usr_delim = ' SYS_USR_DELIM '
c.learning_rate = 0.00005
c.validate_every = 1000
c.batch_size = 2  # FIXME generate batches from train and dev data before training loop
c.dev_batch_size = 1
c.embedding_size=200
c.dropout=1.0
c.rnn_size=600
c.nbest_models=3
c.not_change_limit = 5  # FIXME Be sure that we compare models from different epochs

random.seed(c.seed)
train_set = Dstc2(c.train_file, first_n=10 * c.batch_size)
dev_set = Dstc2(c.dev_file, first_n=3 * c.dev_batch_size)


logger.info('Saving automatically generated stats to config')
c.git_info = git_info()
c.max_seq_len = train_set.turn_len
c.vocab_size = len(train_set.words_vocab)
c.labels_size = len(train_set.labels_vocab)
logger.info('Config\n\n: %s\n\n', c)
logger.info('Saving helper files')
train_set.words_vocab.save(c.vocab_file)
train_set.labels_vocab.save(c.labels_file)
c.save(c.filename)

m = GRUJoint(train_set.turns, train_set.turn_lens, train_set.labels, c)
t = TrainingOps(m.loss, tf.train.GradientDescentOptimizer(c.learning_rate))

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
sess.run(m.turn_initializer, feed_dict={m.turn_initializer: train_set.turns})
sess.run(m.labels_initializer, feed_dict={m.labels_initializer: train_set.labels})
sess.run(m.turn_len_initialzer, feed_dict={m.turn_len_initialzer: train_set.turns_len})

stopper = EarlyStopper(c.nbest_models, c.not_change_limit)
summary_writer = tf.train.SummaryWriter(c.train_dir, sess.graph)
summarize = tf.merge_all_summaries()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


try:
    step = 0
    while not coord.should_stop():
        if step % 100:
            _, step, loss_val, summ_str = sess.run([t.train_op, t.global_step, m.tr_loss, summarize], feed_dict={m.dropout_keep_prob: c.dropbout})
            logger.debug('Step %7d, Loss: %f', step, loss_val)
            summary_writer.add_summary(summ_str, step)
        else:
            _, step = sess.run([t.train_op, t.global_step], feed_dict={m.dropout_keep_prob: c.dropbout})

        if step % c.validate_every == 0:
            true_count, processed = 0, 0
            devlen, devbs = len(dev_set.labels), dev_set.batch_size  # shortcuts
            for i, (b, e) in enumerate(zip(range(0, devlen, devbs), range(devbs, devlen, devbs))):
                turns_val, turns_len_val, labels_val = dev_set.turns[b, e], dev_set.turns_len[b, e], dev_set.labels[b, e]
                true_count += np.sum(sess.run(m.dev_true_count, feed_dict={m.dropout_keep_prob: 1.0,
                    m.feed_turns: turns_val, m.feed_turns_len: turns_len_val, m.feed_labels: labels_val}))
                processed = i * c.dev_batch_size
                logger.debug('Training step: %d, Dev iter: %d, Acc: %f', step, i, true_count / processed)
            dev_acc = true_count / processed
            logger.info('Validation Acc: %.4f, Step: %7d', dev_acc, step)
            if not stopper.save_and_check(dev_acc, step, sess):
                coord.request_stop()
except tf.errors.OutOfRangeError:
    logger.info('Done training interupted by max number of epochs')
finally:
    logger.info('Training stopped after %d steps and %f epochs', step, step / len(train_set))
    coord.request_stop()
coord.join(threads)
sess.close()
