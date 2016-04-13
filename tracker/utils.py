#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter
import json


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
        return dict([(a, getattr(self, a)) for a in dir(self) if not a.startswith('__')])

    def save(self, filename):
        d = self.to_dict()
        rd = {k: v for (k, v) in d.items() if not hasattr(v, '__call__')}
        json.dump(rd, open(filename, 'w'))


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
        self._w2int = dict(((w, i) for w, i in enumerate(tmp)))
        self._int2w = dict(((i, w) for w, i in enumerate(tmp)))
        print(self._w2int)

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
        json.dump(self.__repr__(), open(filename, 'w'))

    def __len__(self):
        return len(self._w2int)
