#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.control_flow_ops import cond
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits as x_entropy
import numpy as np
from datetime import datetime
import logging, random, argparse, os

from tracker.dataset.dstc2 import Dstc2
from tracker.utils import setup_logging

__author__ = '{Petr Belohlavek, Vojtech Hudecek, Josef Valek and Ondrej Platek}'
seed = 123
random.seed(seed)
tf.set_random_seed(seed)


def next_batch(train_set, batch_size):
    indxs = np.random.permutation(len(train_set) // batch_size).tolist() # shuffling but discarding not fully filled last batch
    for b in indxs:
        dlg = train_set.dialogs[b * batch_size:(b+1) * batch_size, :, :]
        lengths = train_set.turn_lens[b * batch_size:(b+1) * batch_size, :]
        labels = train_set.labels[b * batch_size:(b+1) * batch_size, :]
        mask = train_set.dial_mask[b * batch_size:(b+1) * batch_size, :]
        yield (dlg, lengths, labels, mask)


def stats(train, valid, test):
    logging.info('Stats ===============')
    logging.info('Train classes: %d', np.max(train.labels))
    logging.info('Valid classes: %d', np.max(valid.labels))
    logging.info('Test classes: %d', np.max(test.labels))

    train_dist = np.bincount(np.array(train.labels).flatten())
    train_argmax = np.argmax(train_dist)
    logging.info('Train most frequent class: %d (%f)', train_argmax, train_dist[train_argmax]/float(len(train.labels)))

    valid_dist = np.bincount(np.array(valid.labels).flatten())
    valid_argmax = np.argmax(valid_dist)
    logging.info('Valid most frequent class: %d (%f)', valid_argmax, valid_dist[valid_argmax]/float(len(valid.labels)))

    test_dist = np.bincount(np.array(test.labels).flatten())
    test_argmax = np.argmax(test_dist)
    logging.info('Test most frequent class: %d (%f)', test_argmax, test_dist[test_argmax]/float(len(test.labels)))

    logging.info('/Stats ===============')


def main(c):
    ''' params:
            c: config dictionary
    '''

    # Data ---------------------------------------------------------------------------------------------------
    data_portion = None  # 2 * batch_size
    train_set = Dstc2('data/dstc2/data.dstc2.train.json', sample_unk=0.01, first_n=data_portion)
    valid_set = Dstc2('data/dstc2/data.dstc2.dev.json', first_n=data_portion, sample_unk=0,
                      max_dial_len=train_set.max_dial_len, words_vocab=train_set.words_vocab,
                      labels_vocab=train_set.labels_vocab)
    test_set = Dstc2('data/dstc2/data.dstc2.test.json', first_n=data_portion, sample_unk=0,
                     max_dial_len=train_set.max_dial_len, words_vocab=train_set.words_vocab,
                     labels_vocab=train_set.labels_vocab)

    stats(train_set, valid_set, test_set)

    vocab_size = len(train_set.words_vocab)
    output_dim = max(np.unique(train_set.labels)) + 1
    n_train_batches = len(train_set.dialogs) // c.batch_size

    # Model -----------------------------------------------------------------------------------------------------
    logging.info('Creating model')
    input_bt = tf.placeholder('int32', [c.batch_size, train_set.max_turn_len], name='input')
    turn_lens_b = tf.placeholder('int32', [c.batch_size], name='turn_lens')
    mask_b = tf.placeholder('int32', [c.batch_size], name='dial_mask')
    labels_b = tf.placeholder('int64', [c.batch_size], name='labels')
    onehot_labels_bo = tf.one_hot(indices=labels_b,
                                  depth=output_dim,
                                  on_value=1.0,
                                  off_value=0.0,
                                  axis=-1)
    is_first_turn = tf.placeholder(tf.bool)
    gru = GRUCell(c.hidden_state_dim)

    embeddings_we = tf.get_variable('word_embeddings',
                                    initializer=tf.random_uniform([vocab_size, c.embedding_dim], -1.0, 1.0))
    embedded_input_bte = tf.nn.embedding_lookup(embeddings_we, input_bt)
    dialog_state_before_turn = tf.get_variable('dialog_state_before_turn',
                                               initializer=tf.zeros([c.batch_size, c.hidden_state_dim], dtype='float32'),
                                               trainable=False)

    before_state_bh = cond(is_first_turn,
                           lambda: gru.zero_state(c.batch_size, dtype='float32'),
                           lambda: dialog_state_before_turn)

    inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, train_set.max_turn_len, embedded_input_bte)]

    outputs, state_bh = tf.nn.rnn(cell=gru,
                                  inputs=inputs,
                                  initial_state=before_state_bh,
                                  sequence_length=turn_lens_b,
                                  dtype=tf.float32)

    dialog_state_before_turn.assign(state_bh)
    projection_ho = tf.get_variable('project2labels',
                                    initializer=tf.random_uniform([c.hidden_state_dim, output_dim], -1.0, 1.0))

    logits_bo = tf.matmul(state_bh, projection_ho)
    tf.histogram_summary('logits', logits_bo)

    probabilities_bo = tf.nn.softmax(logits_bo)
    tf.histogram_summary('probabilities', probabilities_bo)

    float_mask_b = tf.cast(mask_b, 'float32')
    loss = tf.reduce_sum(tf.mul(float_mask_b, x_entropy(logits_bo, onehot_labels_bo))) / tf.reduce_sum(float_mask_b)
    tf.scalar_summary('CCE loss', loss)

    predict_b = tf.argmax(logits_bo, 1)
    correct = tf.cast(tf.equal(predict_b, labels_b), 'float32')
    accuracy = tf.reduce_sum(tf.mul(correct, float_mask_b)) / tf.reduce_sum(float_mask_b)
    tf.scalar_summary('Accuracy', accuracy)

    tb_info = tf.merge_all_summaries()

    # Optimizer  -----------------------------------------------------------------------------------------------------
    logging.info('Creating optimizer')
    optimizer = tf.train.AdamOptimizer(c.learning_rate)
    logging.info('Creating train_op')
    train_op = optimizer.minimize(loss)

    # Session  -----------------------------------------------------------------------------------------------------
    logging.info('Creating session')
    sess = tf.Session()
    logging.info('Initing variables')
    init = tf.initialize_all_variables()
    logging.info('Running session')
    sess.run(init)

    # TB ---------------------------------------------------------------------------------------------------------
    logging.info('See stats via tensorboard: $ tensorboard --logdir %s', c.log_dir)
    train_writer = tf.train.SummaryWriter(c.log_dir, sess.graph)

    # Train ---------------------------------------------------------------------------------------------------------
    train_summary = None
    for e in range(c.epochs):
        logging.info('------------------------------')
        logging.info('Epoch %d', e)

        total_loss = 0
        total_acc = 0
        batch_count = 0
        for bid, (dialogs_bTt, lengths_bT, labels_bT, masks_bT) in enumerate(next_batch(train_set, c.batch_size)):
            turn_loss = 0
            turn_acc = 0
            n_turns = 0
            first_run = True
            for (turn_bt, label_b, lengths_b, masks_b) in zip(dialogs_bTt.transpose([1, 0, 2]),
                                                              labels_bT.transpose([1, 0]),
                                                              lengths_bT.transpose([1, 0]),
                                                              masks_bT.transpose([1,0])):
                if sum(masks_b) == 0:
                    break 

                _, batch_loss, batch_accuracy, train_summary = sess.run([train_op, loss, accuracy, tb_info],
                                                                        feed_dict={input_bt: turn_bt,
                                                                                   turn_lens_b: lengths_b,
                                                                                   mask_b: masks_b,
                                                                                   labels_b: label_b,
                                                                                   is_first_turn: first_run})
                first_run = False
                turn_loss += batch_loss
                turn_acc += batch_accuracy
                n_turns += 1

            total_loss += turn_loss / n_turns
            total_acc += turn_acc / n_turns
            batch_count += 1

            logging.info('Batch %d/%d\r', bid, n_train_batches)

        train_writer.add_summary(train_summary, e)
        logging.info('Train cost %f', total_loss / batch_count)
        logging.info('Train accuracy: %f', total_acc / batch_count)

        def monitor_stream(work_set, name):
            total_loss = 0
            total_acc = 0
            n_valid_batches = 0
            for bid, (dialogs_bTt, lengths_bT, labels_bT, masks_bT) in enumerate(next_batch(work_set, c.batch_size)):
                turn_loss = 0
                turn_acc = 0
                n_turns = 0
                first_run = True
                for (turn_bt, label_b, lengths_b, masks_b) in zip(dialogs_bTt.transpose([1, 0, 2]),
                                                                  labels_bT.transpose([1, 0]),
                                                                  lengths_bT.transpose([1, 0]),
                                                                  masks_bT.transpose([1, 0])):
                    if sum(masks_b) == 0:
                        break

                    input = np.pad(turn_bt, ((0, 0), (0, train_set.max_turn_len-turn_bt.shape[1])),
                                   'constant', constant_values=0) if train_set.max_turn_len > turn_bt.shape[1]\
                        else turn_bt

                    predictions, batch_loss, batch_acc, valid_summary = sess.run([predict_b, loss, accuracy, tb_info],
                                                                                 feed_dict={input_bt: input,
                                                                                            turn_lens_b: lengths_b,
                                                                                            labels_b: label_b,
                                                                                            mask_b: masks_b,
                                                                                            is_first_turn: first_run})
                    turn_loss += batch_loss
                    turn_acc += batch_acc
                    first_run = False
                    n_turns += 1

                total_loss += turn_loss / n_turns
                total_acc += turn_acc / n_turns
                n_valid_batches += 1

            logging.info('%s cost: %f', name, total_loss/n_valid_batches)
            logging.info('%s accuracy: %f', name, total_acc/n_valid_batches)

        monitor_stream(valid_set, 'Valid')
        monitor_stream(test_set, 'Test')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--learning_rate', type=float, default=0.005)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--hidden_state_dim', type=int, default=200)
    ap.add_argument('--embedding_dim', type=int, default=300)
    ap.add_argument('--log_dir', default=None)
    ap.add_argument('--exp', default='exp', help='human readable experiment name')
    c = ap.parse_args()
    c.name = 'log/%(u)s-%(n)s/%(u)s%(n)s' % {'u': datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S.%f')[:-3], 'n': c.exp}
    c.log_name = '%s.log' % c.name
    c.log_dir = c.log_dir or c.name + 'tensorboard'

    os.makedirs(os.path.dirname(c.name), exist_ok=True)
    setup_logging(c.log_name)
    logging.info('Start with config: %s', c)
    main(c)
    logging.info('Finished')
