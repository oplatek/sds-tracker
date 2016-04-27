#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for dialogue state tracking.

"""
import logging, uuid, random
import tensorflow as tf
from tracker.utils import Config, git_info, compare_ref, setup_logging
from tracker.model import GRUJoint
from tracker.training import TrainingOps, EarlyStopper
from tracker.dataset.dstc2 import Dstc2


c = Config()  # FIXME load config from json_file after few commints when everybody knows the settings
c.name='log/demo_%s' % uuid.uuid1()
c.train_dir = c.name + '_traindir'
c.filename = '%s.json' % c.name
c.vocab_file = '%s.vocab' % c.name
c.labels_file = '%s.labels' % c.name
c.log_name = '%s.log' % c.name
c.seed=123
c.train_file = './data/dstc2/data.dstc2.train.json'
c.dev_file = './data/dstc2/data.dstc2.dev.json'
c.epochs = 2
c.sys_usr_delim = ' SYS_USR_DELIM '
c.learning_rate = 0.00005
c.validate_every = 1
c.train_sample_every = 1
c.batch_size = 2
c.dev_batch_size = 1
c.embedding_size=200
c.dropout=1.0
c.rnn_size=600
c.nbest_models=3
c.not_change_limit = 5  # FIXME Be sure that we compare models from different epochs
c.sample_unk = 3

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
train_set.words_vocab.save(c.vocab_file)
train_set.labels_vocab.save(c.labels_file)
c.save(c.filename)

m = GRUJoint(train_set.dialogs, train_set.turn_lens, train_set.labels, c)
t = TrainingOps(m.tr_loss, tf.train.GradientDescentOptimizer(c.learning_rate))

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
sess.run(m.turn_inputs.initializer, feed_dict={m.turn_initializer: train_set.dialogs})
sess.run(m.turn_lengths.initializer, feed_dict={m.turn_len_initialzer: train_set.turn_lens})
sess.run(m.input_labels.initializer, feed_dict={m.labels_initializer: train_set.labels})

stopper = EarlyStopper(c.nbest_models, c.not_change_limit, c.name)
logger.info('Monitor progress by tensorboard:\ntensorboard --logdir "%s"\n', c.train_dir)
summary_writer = tf.train.SummaryWriter(c.train_dir, sess.graph)
summarize = tf.merge_all_summaries()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


try:
    step = 0
    while not coord.should_stop():
        if step % c.train_sample_every == 0:
            _, step, loss_val, tr_inputs, tr_lab, tr_pred_val, summ_str = sess.run([t.train_op, t.global_step, m.tr_loss, 
                m.turn_inputs, m.input_labels, m.tr_predictss, summarize], feed_dict={m.dropout_keep_prob: c.dropout})
            logger.debug('Step %7d, Loss: %f', step, loss_val)
            logger.debug('Step %7d input example:\n%s', step, compare_ref(tr_inputs, tr_lab, tr_pred_val, train_set.words_vocab, train_set.labels_vocab))
            summary_writer.add_summary(summ_str, step)
        else:
            _, step = sess.run([t.train_op, t.global_step], feed_dict={m.dropout_keep_prob: c.dropout})

        if step % c.validate_every == 0:
            tot_true_count, tot_num_turn = 0, 0
            devlen, devbs = len(dev_set), c.dev_batch_size  # shortcuts
            for i, (b, e) in enumerate(zip(range(0, devlen, devbs), range(devbs, devlen, devbs))):  # FIXME omit the last batch
                dials_val, turn_lens_val, labels_val = dev_set.dialogs[b:e, :], dev_set.turn_lens[b:e], dev_set.labels[b:e]
                true_count, num_turn = sess.run(m.dev_true_count, m.dev_dial_len, feed_dict={m.dropout_keep_prob: 1.0,
                    m.feed_dials: dials_val, m.feed_turn_lens: turn_lens_val, m.feed_labels: labels_val})
                tot_true_count, tot_num_turn = tot_true_count + true_count, tot_num_turn + num_turn
                logger.debug('Training step: %d, Dev iter: %d, Batch Acc: %f', step, i, true_count / num_turn)

            dev_acc = tot_true_count / tot_num_turn
            logger.info('Validation Acc: %.4f, Step: %7d', dev_acc, step) 

            dev_pred_val = sess.run(m.dev_predictss, feed_dict={m.dropout_keep_prob: 1.0,
                    m.feed_dials: dials_val, m.feed_turn_lens: turn_lens_val, m.feed_labels: labels_val})
            logger.debug('Step %7d dev last batch example:\n%s\n', step, compare_ref(dials_val, labels_val, dev_pred_val, train_set.words_vocab, train_set.labels_vocab))

            if not stopper.save_and_check(dev_acc, step, sess):
                coord.request_stop()
except tf.errors.OutOfRangeError:
    logger.info('Done training interupted by max number of epochs')
finally:
    logger.info('Training stopped after %7d steps and %7.2f epochs', step, step / len(train_set))
    coord.request_stop()
coord.join(threads)
sess.close()
