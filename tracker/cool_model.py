import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.functional_ops import scan
from tensorflow.python.ops.control_flow_ops import cond

import numpy as np
import logging

from tracker.dataset.dstc2 import Dstc2

__author__ = 'Petr Belohlavek, Vojtech Hudecek, Josef Valek'


def next_batch(train_set, batch_size):
    b = 0
    while 1:
        dlg, labels = train_set.dialogs[b * batch_size:(b+1) * batch_size, :, :], train_set.labels[b * batch_size:(b+1) * batch_size, :]
        b += 1
        if dlg.shape[0] < 1:
            break
        yield (dlg, labels)


def main():
    # Config -----------------------------------------------------------------------------------------------------
    learning_rate = 0.005
    batch_size = 32
    epochs = 10
    hidden_state_dim = 200
    embedding_dim = 300
    log_dir = 'log'

    # Data ---------------------------------------------------------------------------------------------------
    train_set = Dstc2('data/dstc2/data.dstc2.train.json', sample_unk=0, first_n=2 * batch_size)
    vocab_size = len(train_set.words_vocab)
    output_dim = max(np.unique(train_set.labels)) + 1
    n_train_batches = len(train_set.dialogs) // batch_size

    # Model -----------------------------------------------------------------------------------------------------
    logging.info('Creating model')
    input_bt = tf.placeholder('int32', [batch_size, train_set.max_turn_len], name='input')
    labels_b = tf.placeholder('int64', [batch_size], name='labels')
    onehot_labels_bo = tf.one_hot(indices=labels_b,
                                  depth=output_dim,
                                  on_value=1.0,
                                  off_value=0.0,
                                  axis=-1)
    is_first_turn = tf.placeholder(tf.bool)

    gru = GRUCell(hidden_state_dim)
    embeddings_we = tf.get_variable('word_embeddings', initializer=tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0))
    embedded_input_bte = tf.nn.embedding_lookup(embeddings_we, input_bt)
    dialog_state_before_turn = tf.get_variable('dialog_state_before_turn', initializer=tf.zeros([batch_size, hidden_state_dim], dtype='float32'), trainable=False)

    before_state_bh = cond(is_first_turn,
        lambda: gru.zero_state(batch_size, dtype='float32'),
        lambda: dialog_state_before_turn)

    state_tbh = scan(fn=lambda last_state_bh, curr_input_bte: gru(curr_input_bte, last_state_bh)[1],
                    elems=tf.transpose(embedded_input_bte, perm=[1, 0, 2]),
                    initializer=before_state_bh)

    state_bh = state_tbh[state_tbh.get_shape()[0]-1, :, :]
    dialog_state_before_turn.assign(state_bh)

    projection_ho = tf.get_variable('project2labels',
                                    initializer=tf.random_uniform([hidden_state_dim, output_dim], -1.0, 1.0))

    logits_bo = tf.matmul(state_bh, projection_ho)
    tf.histogram_summary('logits', logits_bo)

    probabilities_bo = tf.nn.softmax(logits_bo)
    tf.histogram_summary('probabilities', probabilities_bo)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_bo, onehot_labels_bo))

    tf.scalar_summary('CCE loss', loss)

    predict_b = tf.argmax(logits_bo, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_b, labels_b), 'float32'), name='accuracy')
    tf.scalar_summary('Accuracy', accuracy)
    tb_info = tf.merge_all_summaries()

    # Optimizer  -----------------------------------------------------------------------------------------------------
    logging.info('Creating optimizer')
    optimizer = tf.train.AdamOptimizer(learning_rate)
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
    logging.info('See stats via tensorboard: $ tensorboard --logdir %s', log_dir)
    train_writer = tf.train.SummaryWriter(log_dir, sess.graph)

    # Train ---------------------------------------------------------------------------------------------------------
    train_summary = None
    n_train_batches=10
    for e in range(epochs):
        epoch_loss = 0
        for bid, (dialogs_bTt, labels_bT) in enumerate(next_batch(train_set, batch_size)):
            first_run = True
            for turn_bt, label_b in zip(dialogs_bTt.transpose([1,0,2]), labels_bT.transpose([1,0])):
                _, batch_loss, train_summary = sess.run([train_op, loss, tb_info], feed_dict={input_bt: turn_bt,
                                                                                              labels_b: label_b,
                                                                                              is_first_turn:first_run})
                first_run = False
                epoch_loss += batch_loss

            if bid % 10 == 0:
                logging.info('Batch %d/%d', bid, n_train_batches)

        train_writer.add_summary(train_summary, e)

        logging.info('Average Batch Loss @%d = %f', e, epoch_loss/n_train_batches)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.info('Start the script from parent directory by running "python tracker/cool_model.py')
    main()
    logging.info('Finished')
