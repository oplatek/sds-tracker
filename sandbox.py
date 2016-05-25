# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import json, logging, uuid, random
# import tensorflow as tf
# import numpy as np
# from tracker.utils import Config, Vocabulary, pad, labels2words, git_info
# from tracker.model import GRUJoint
# from tracker.fat_model import FatModel
# from tensorflow.python.ops.functional_ops import scan
#
#
# if __name__ == '__main__':
#
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#    # FUN
#
#     # x_ab = tf.placeholder("float32", [2,3], name='x')
#     # y_b = tf.placeholder("float32", [3], name='y')
#     #
#     # z = tf.tile(tf.expand_dims(y_b, 0), [x_ab.get_shape()[0].value, 1])
#
#     x_aa = tf.placeholder('float32', [5,5], 'x')
#     y_a = tf.placeholder('float32', [5], name='y')
#
#     res_a = scan(fn=lambda last_output_a, current_input_a: last_output_a + current_input_a,
#                  elems=x_aa,
#                  initializer=y_a)
#
#     sess = tf.Session()
#     init = tf.initialize_all_variables()
#     sess.run(init)
#
#     res = sess.run([res_a], feed_dict={x_aa: np.eye(5, dtype='float32')),
#                                        y_a: np.random.zeros(5)})
#     print(res)

############################################################
# from __future__ import division, print_function
# import tensorflow as tf
# from tensorflow.python.ops import functional_ops
#
#
# def fn(previous_output, current_input):
#     return previous_output + current_input
#
# elems = tf.Variable([1.0, 2.0, 2.0, 2.0])
# elems = tf.identity(elems)
# initializer = tf.constant(0.0)
# out = functional_ops.scan(fn, elems, initializer=initializer)
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(out))
############################################################
from __future__ import division, print_function
import tensorflow as tf
from tensorflow.python.ops import functional_ops


def fn(previous_output, a, b):
    return previous_output + a*b

A = tf.Variable([1.0, 2.0, 3.0])
B = tf.Variable([1.0, -2.0, 5.0])
A = tf.identity(A)
B = tf.identity(B)

initA = tf.constant(0.0)
initB = tf.constant(0.0)

out = functional_ops.scan(fn, tf.identity([A,B]), initializer=tf.identity([initA, initB]))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(out))