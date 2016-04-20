#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, logging, uuid, random
import tensorflow as tf
import numpy as np
from tracker.utils import Config, Vocabulary, pad, labels2words, git_info
from tracker.model import GRUJoint
from tracker.fat_model import FatModel

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
   # FUN

    x_ab = tf.placeholder("float32", [2,3], name='x')
    y_b = tf.placeholder("float32", [3], name='y')

    z = tf.tile(tf.expand_dims(y_b, 0), [x_ab.get_shape()[0].value, 1])

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    zz = sess.run([z], feed_dict={x_ab: np.zeros((2,3)),
                                  y_b: np.random.rand(3)})
    print(zz[0].shape)
    print(zz[0])
