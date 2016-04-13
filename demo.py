#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, logging, uuid, random
import tensorflow as tf
import numpy as np
from tracker.utils import Config, Vocabulary, pad, labels2words
from tracker.model import GRUJoint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

c = Config()  # FIXME load config from json_file after few commints when everybody knows the settings
c.seed=123
c.train_file = './data.dstc2.train.json'
c.dev_file = './data.dstc2.dev.json'
c.epochs = 2
c.config_file = 'log/demo_%s' % uuid.uuid1()
c.vocab_file = '%s.vocab' % c.config_file
c.labels_file = '%s.labels' % c.config_file
c.sys_usr_delim = ' SYS_USR_DELIM '
c.learning_rate = 0.00005
c.max_seq_len = 60
c.validate_every = 1
c.batch_size = 1  # FIXME generate batches from train and dev data before training loop
random.seed(c.seed)
train = json.load(open(c.train_file))
dev = json.load(open(c.dev_file))

# words apearing in system utterances, user utterances plus delimiter between the two
vocab = Vocabulary([w for dialog in train for turn in dialog for w in (turn[0] + turn[1]).split()], 
        extra_words=[c.sys_usr_delim])
# join slots as string e.g. "indian east moderate"
labels = Vocabulary([turn[4] for dialog in train for turn in dialog])

c.vocab_size = len(vocab)
c.labels_size = len(labels)

m = GRUJoint(c)
optimizer = tf.train.GradientDescentOptimizer(c.learning_rate)
train_op = optimizer.minimize(m.loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


for e in range(c.epochs):
    skipped_dial = 0
    for d_i, dialog in enumerate(train):
        for turn in dialog:
            sys_utt, usr_utt, usr_asr, asr_score, slots = turn
            input_val = pad([vocab.get_i(w) for w in (sys_utt + c.sys_usr_delim + usr_utt).split()], c.max_seq_len)
            labels_val = np.array(labels.get_i(slots)); labels_val.shape = (1, 1)
            if len(input_val) > c.max_seq_len:
                skipped_dial += 1
                break  # stop processing the rest of dialog
            _, loss = sess.run([m.train_op, m.loss], feed_dict={m.input: input_val, m.labels: labels_val})
            logger.debug('Sys: %s\nUsr: %s\nSlots: %s\nLoss: %f\n', sys_utt, usr_utt, slots, loss)
    random.shuffle(train)  # fix the seed
    logger.info('Train epoch %d: %d / %d interrupted', e, skipped_dial, len(train)) 
    if e % c.validate_every == 0:
        accs = np.empty(shape=(0, 1))
        for d_i, dialog in enumerate(dev):
            for turn in dialog:
                sys_utt, usr_utt, usr_asr, asr_score, slots = turn
                input_val = pad([vocab.get_i(w) for w in (sys_utt + c.sys_usr_delim + usr_utt).split()], c.max_seq_len)
                labels_val = np.array(labels.get_i(slots)); labels_val.shape = (1, 1)
                y, loss, acc = sess.run([m.predict, m.loss, m.acc], feed_dict={m.input: input_val, m.labels: labels_val})
                accs = np.vstack((accs, acc))
                logger.debug('Sys: %s\nUsr: %s\nSlots: %s\nSlotsY: %s\nAcc: %d\nLoss: %d', sys_utt, usr_utt, slots, labels2words(y, labels), acc, loss)
        logger.info('Accuracy for epoch %d: %f', e, np.mean(accs, axis=0))

vocab.save(c.vocab_file)
labels.save(c.labels_file)
c.save(c.config_file)
