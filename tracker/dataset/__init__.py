#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from collections import Counter
import random


__all__ = ['dstc.Dstc2']


class TurnTrackerSet(object):
    '''
    Encodes a dialogue dataset into array of turns,
    where turn are represented by the same fixed length vectors.
    '''
    @property
    def turns(self):
        '''
        Returns list of fixed lenght numpy arrays encoding the
        input_bdt of a turn.
        Content of the numpy arrays are integers representing words.
        We suggest to padd the not used values with zeros.
        '''
        raise NotImplementedError('Implement in derived class!')

    @property
    def turn_lens(self):
        '''
        Returns lenghts of turns in words.
        '''
        raise NotImplementedError('Implement in derived class!')

    @property
    def labels(self):
        '''
        Returns list labels_bd
        '''
        raise NotImplementedError('Implement in derived class!')

    @property 
    def words_vocab(self):
        '''Returns Vocabulary for words in turns'''
        raise NotImplementedError('Implement in derived class!')

    @property 
    def labels_vocab(self):
        '''Returns Vocabulary for predicted labels_bd'''
        raise NotImplementedError('Implement in derived class!')


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
        tmp = set(list(self.extra_words) + [w for w, _ in self._counts.most_common(max_items)])
        self._w2int = dict(((w, i) for i, w in enumerate(tmp)))
        self._int2w = dict(((i, w) for i, w in enumerate(tmp)))

    def get_i(self, w, unk_chance_smaller=0):
        if unk_chance_smaller:
            wc = self._counts[w]
            if wc <= unk_chance_smaller and wc > random.uniform(0, unk_chance_smaller):
                return self._w2int[self.unk]
            else:
                return self._w2int.get(w, self._w2int[self.unk])
        else:
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
