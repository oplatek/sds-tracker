#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import subprocess
import logging
import tensorflow as tf
import heapq


logger = logging.getLogger(__name__)


class Config:
    @classmethod
    def load_json(cls, filename):
        json_obj = json.load(open(filename), 'r')
        return cls.from_dict(json_obj)

    @classmethod
    def from_dict(cls, d):
        c = cls()
        for k, v in d.items():
            c.k = v
        return d

    def to_dict(self):
        d = dict([(a, getattr(self, a)) for a in dir(self) if not a.startswith('__')])
        stripped_d = {k: v for (k, v) in d.items() if not hasattr(v, '__call__')}
        return stripped_d

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def save(self, filename=None):
        json.dump(self.to_dict(), open(filename, 'w'), indent=4, sort_keys=True)


def git_info():
    head, diff, remote = None, None, None
    try:
        head = subprocess.getoutput('git rev-parse HEAD').strip()
    except subprocess.CalledProcessError:
        pass
    try:
        diff = subprocess.getoutput('git diff --no-color')
    except subprocess.CalledProcessError:
        pass
    try:
        remote = subprocess.getoutput('git remote -v').strip()
    except subprocess.CalledProcessError:
        pass
    git_dict = {'head': head or 'Unknown',
            'diff': diff or 'Unknown',
            'remote': remote or 'Unknown'}
    return git_dict


def compare_ref(inputs, labels, predictss, vocab, labelsdict):
    inputs = np.reshape(inputs, (np.prod(inputs.shape[:-1]), inputs.shape[-1]))
    pred_idx = np.reshape(predictss, (np.prod(predictss.shape),)).tolist()
    labels = np.reshape(labels, (np.prod(labels.shape),)).tolist()
    res = []
    for i, (pr, lb) in enumerate(zip(pred_idx, labels)):
        turn = 'turn: ' + ' '.join([vocab.get_w(k) for k in inputs[i, :].tolist()])
        lbs= 'label: ' + labelsdict.get_w(lb)
        prs = 'predict: ' + labelsdict.get_w(pr)
        res.extend([turn, lbs, prs])
    return '\n'.join(res)


def setup_logging(filename, console_level=logging.INFO):
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG, filename=filename)
    if isinstance(console_level, str):
        console_level = logging.getLevelName(console_level)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    logging.getLogger('').addHandler(console)


class EarlyStopper(object):
    '''Keeping track of n_best highest values in reward'''
    def __init__(self, track_n_best, not_change_limit, saver_prefix):
        self.n_best = track_n_best
        self.not_change_limit = not_change_limit
        self._heap = []
        self._not_improved = 0
        self.saver = tf.train.Saver()
        self.saver_prefix = saver_prefix

    @property
    def rewards_steps_sessions(self):
        '''Returns n_best results sorted from the highest to the smallest.'''
        return reversed([heapq.heappop(self._heap) for i in range(len(self._heap))])

    def save_and_check(self, reward, step, sess):

        def save(reward, step):
            path = self.saver.save(sess, '%s-reward-%.4f-step-%07d' % (self.saver_prefix, reward, step))
            logger.info('Sess: %f saved to %s', reward, path)
            return path

        if len(self._heap) < self.n_best:
            self._not_improved = 0
            path = save(reward, step)
            heapq.heappush(self._heap, (reward, step, path))
        else:
            last_reward = self._heap[0][0]
            if last_reward < reward:
                heapq.heappop(self._heap)
                path = save(reward, step)
                heapq.heappush(self._heap, (reward, step, path))
                self._not_improved = 0
            else:
                logger.info('Not keeping reward %f from step %d', reward, step)
                self._not_improved += 1
        return self._not_improved <= self.not_change_limit

    def highest_reward(self):
        return max(self._heap)
