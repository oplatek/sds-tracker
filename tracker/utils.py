#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import subprocess
import logging


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


def compare_ref(inputs, labels, logits, vocab, labelsdict):
    res = []
    pred_idx = np.argmax(logits, axis=1)
    for i, (lb, pr) in enumerate(zip(labels.tolist(), pred_idx.tolist())):
        turn = 'turn: ' + ' '.join([vocab.get_w(k) for k in inputs[i, :].tolist()])
        lbs= 'label: ' + labelsdict.get_w(lb)
        prs = 'predict: ' + labelsdict.get_w(pr)
        res.extend([turn, lbs, prs])
    return '\n'.join(res)


def setup_logging(filename):
    logging.basicConfig(level=logging.DEBUG, filename=filename)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
