#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter
import json
import numpy as np
import subprocess


# FIXME rewrite for batch
def pad(arr, max_len):
    to_pad = max_len - len(arr)
    arr = np.array(arr + ([0] * to_pad))
    arr.shape = (1, max_len)
    return arr


def labels2words(lbl, vocab):
    lables = []
    for i in range(lbl.shape[0]):
        lables.append([vocab.get_w(k) for k in lbl[i, :]])
    return lables


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


class Vocabulary:
    @classmethod
    def load_json(cls, filename):
        d = json.load(open(filename, 'r'))
        return cls(**d)

    def __init__(self, counts, max_items=5000, extra_words=None, unk='UNK'):
        self._counts = counts if isinstance(counts, Counter) else Counter(counts)
        self.extra_words = set(extra_words or [])
        self.extra_words.add(unk)
        self.unk = unk
        self.max_items = max_items
        tmp = list(self.extra_words) + [w for w, _ in self._counts.most_common(max_items)]
        self._w2int = dict(((w, i) for i, w in enumerate(tmp)))
        self._int2w = dict(((i, w) for i, w in enumerate(tmp)))

    def get_i(self, w):
        return self._w2int.get(w, self._w2int[self.unk])

    def get_w(self, index):
        return self._int2w[index]

    def __repr__(self):
        return {'counts': self._counts,
                'max_items': self.max_items,
                'unk': self.unk,
                'extra_words': list(self.extra_words)}

    def save(self, filename):
        json.dump(self.__repr__(), open(filename, 'w'), indent=4, sort_keys=True)

    def __len__(self):
        return len(self._w2int)


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
