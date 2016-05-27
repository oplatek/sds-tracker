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
        labels = train_set.labels_separate[b * batch_size:(b+1) * batch_size, :, :]
        mask = train_set.dial_mask[b * batch_size:(b+1) * batch_size, :]
        yield (dlg, lengths, labels[:,:,0], labels[:,:,1], labels[:,:,2], mask)


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


def get_labels_with_onehot(batch_size, output_dim, name):
    labels = tf.placeholder('int64', [batch_size], name=name)
    onehot_labels = tf.one_hot(indices=labels,
                                  depth=output_dim,
                                  on_value=1.0,
                                  off_value=0.0,
                                  axis=-1)
    return labels, onehot_labels

def get_logits_and_probabilities(state_bh, hidden_state_dim, output_dim, name):
    projection = tf.get_variable('projection2' + name,
                            initializer=tf.random_uniform([hidden_state_dim, output_dim], -1.0, 1.0))

    logits = tf.matmul(state_bh, projection)
    tf.histogram_summary('logits_'+name, logits)

    probabilities = tf.nn.softmax(logits)
    tf.histogram_summary('probabilities'+name, probabilities)

    predictions = tf.argmax(logits, 1)

    return logits, probabilities, predictions

def get_accuracy(correct, mask):
    return tf.reduce_sum(tf.mul(correct, mask)) / tf.reduce_sum(mask)

def main(c):
    ''' params:
            c: config dictionary
    '''

    # Data ---------------------------------------------------------------------------------------------------
    data_portion = None  # 2 * batch_size
    train_set = Dstc2('data/dstc2/data.dstc2.train.json', sample_unk=0.01, first_n=data_portion)
    valid_set = Dstc2('data/dstc2/data.dstc2.dev.json', first_n=data_portion, sample_unk=0,
                      max_dial_len=train_set.max_dial_len, words_vocab=train_set.words_vocab,
                      labels_vocab=train_set.labels_vocab, labels_vocab_separate=train_set.labels_vocab_separate)
    test_set = Dstc2('data/dstc2/data.dstc2.test.json', first_n=data_portion, sample_unk=0,
                     max_dial_len=train_set.max_dial_len, words_vocab=train_set.words_vocab,
                     labels_vocab=train_set.labels_vocab, labels_vocab_separate=train_set.labels_vocab_separate)

    stats(train_set, valid_set, test_set)

    vocab_size = len(train_set.words_vocab)
    output_dim = max(np.unique(train_set.labels)) + 1
    n_train_batches = len(train_set.dialogs) // c.batch_size

    # output dimensions for each separate label
    output_dims = []
    for i in range(3):
        o_d = max(np.unique(train_set.labels_separate[:,:,i])) + 1
        output_dims.append(o_d)


    # Model -----------------------------------------------------------------------------------------------------
    logging.info('Creating model')
    input_bt = tf.placeholder('int32', [c.batch_size, train_set.max_turn_len], name='input')
    turn_lens_b = tf.placeholder('int32', [c.batch_size], name='turn_lens')
    mask_b = tf.placeholder('int32', [c.batch_size], name='dial_mask')
    # labels_b = tf.placeholder('int64', [c.batch_size], name='labels')
    # onehot_labels_bo = tf.one_hot(indices=labels_b,
    #                               depth=output_dim,
    #                               on_value=1.0,
    #                               off_value=0.0,
    #                               axis=-1)

    # separate labels and their onehots
    labels0_b, onehot_labels0_bo0 = get_labels_with_onehot(c.batch_size, output_dims[0], 'labels0')
    labels1_b, onehot_labels1_bo1 = get_labels_with_onehot(c.batch_size, output_dims[1], 'labels1')
    labels2_b, onehot_labels2_bo2 = get_labels_with_onehot(c.batch_size, output_dims[2], 'labels2')

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


    # projection_ho = tf.get_variable('project2labels',
    #                                 initializer=tf.random_uniform([c.hidden_state_dim, output_dim], -1.0, 1.0))



    # logits_bo = tf.matmul(state_bh, projection_ho)
    # tf.histogram_summary('logits', logits_bo)

    # probabilities_bo = tf.nn.softmax(logits_bo)
    # tf.histogram_summary('probabilities', probabilities_bo)


    # logits and probabilites and predictions from hidden state
    logits_bo0, probabilities_bo0, predict_b0 = get_logits_and_probabilities(state_bh, c.hidden_state_dim, output_dims[0], 'labels0')
    logits_bo1, probabilities_bo1, predict_b1 = get_logits_and_probabilities(state_bh, c.hidden_state_dim, output_dims[1], 'labels1')
    logits_bo2, probabilities_bo2, predict_b2 = get_logits_and_probabilities(state_bh, c.hidden_state_dim, output_dims[2], 'labels2')



    float_mask_b = tf.cast(mask_b, 'float32')
    
    # loss = tf.reduce_sum(tf.mul(float_mask_b, x_entropy(logits_bo, onehot_labels_bo))) / tf.reduce_sum(float_mask_b)
    # tf.scalar_summary('CCE loss', loss)

    # losses
    loss_0 = tf.reduce_sum(tf.mul(float_mask_b, x_entropy(logits_bo0, onehot_labels0_bo0))) / tf.reduce_sum(float_mask_b)
    loss_1 = tf.reduce_sum(tf.mul(float_mask_b, x_entropy(logits_bo1, onehot_labels1_bo1))) / tf.reduce_sum(float_mask_b)
    loss_2 = tf.reduce_sum(tf.mul(float_mask_b, x_entropy(logits_bo2, onehot_labels2_bo2))) / tf.reduce_sum(float_mask_b)
    loss = loss_0 + loss_1 + loss_2
    tf.scalar_summary('CCE loss', loss)


    # predict_b = tf.argmax(logits_bo, 1)
    # correct = tf.cast(tf.equal(predict_b, labels_b), 'float32')
    # accuracy = tf.reduce_sum(tf.mul(correct, float_mask_b)) / tf.reduce_sum(float_mask_b)
    # tf.scalar_summary('Accuracy', accuracy)


    # correct
    correct_0 = tf.cast(tf.equal(predict_b0, labels0_b), 'float32')
    correct_1 = tf.cast(tf.equal(predict_b1, labels1_b), 'float32')
    correct_2 = tf.cast(tf.equal(predict_b2, labels2_b), 'float32')
    correct_all = tf.mul(tf.mul(correct_0, correct_1), correct_2)

    # accuracies
    accuracy_0 = get_accuracy(correct_0, float_mask_b)
    accuracy_1 = get_accuracy(correct_1, float_mask_b)
    accuracy_2 = get_accuracy(correct_2, float_mask_b)
    accuracy_all = get_accuracy(correct_all, float_mask_b)
    tf.scalar_summary('Accuracy all', accuracy_all)    
    tf.scalar_summary('Accuracy label 0', accuracy_0)
    tf.scalar_summary('Accuracy label 1', accuracy_1)
    tf.scalar_summary('Accuracy label 2', accuracy_2)



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
        for bid, (dialogs_bTt, lengths_bT, labels0_bT, labels1_bT, labels2_bT, masks_bT) in enumerate(next_batch(train_set, c.batch_size)):
            turn_loss = 0
            turn_acc = 0
            n_turns = 0
            first_run = True
            for (turn_bt, label0_b, label1_b, label2_b, lengths_b, masks_b) in zip(dialogs_bTt.transpose([1, 0, 2]),
                                                              labels0_bT.transpose([1, 0]),
                                                              labels1_bT.transpose([1, 0]),
                                                              labels2_bT.transpose([1, 0]),
                                                              lengths_bT.transpose([1, 0]),
                                                              masks_bT.transpose([1,0])):
                if sum(masks_b) == 0:
                    break 

                _, batch_loss, batch_accuracy, train_summary = sess.run([train_op, loss, accuracy_all, tb_info],
                                                                        feed_dict={input_bt: turn_bt,
                                                                                   turn_lens_b: lengths_b,
                                                                                   mask_b: masks_b,
                                                                                   labels0_b: label0_b,
                                                                                   labels1_b: label1_b,
                                                                                   labels2_b: label2_b,
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
            for bid, (dialogs_bTt, lengths_bT, labels0_bT, labels1_bT, labels2_bT, masks_bT) in enumerate(next_batch(work_set, c.batch_size)):
                turn_loss = 0
                turn_acc = 0
                n_turns = 0
                first_run = True
                for (turn_bt, label0_b, label1_b, label2_b, lengths_b, masks_b) in zip(dialogs_bTt.transpose([1, 0, 2]),
                                                              labels0_bT.transpose([1, 0]),
                                                              labels1_bT.transpose([1, 0]),
                                                              labels2_bT.transpose([1, 0]),
                                                              lengths_bT.transpose([1, 0]),
                                                              masks_bT.transpose([1,0])):
                    if sum(masks_b) == 0:
                        break

                    input = np.pad(turn_bt, ((0, 0), (0, train_set.max_turn_len-turn_bt.shape[1])),
                                   'constant', constant_values=0) if train_set.max_turn_len > turn_bt.shape[1]\
                        else turn_bt

                    batch_loss, batch_acc, valid_summary = sess.run([loss, accuracy_all, tb_info],
                                                                                 feed_dict={input_bt: input,
                                                                                            turn_lens_b: lengths_b,
                                                                                           labels0_b: label0_b,
                                                                                           labels1_b: label1_b,
                                                                                           labels2_b: label2_b,
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
