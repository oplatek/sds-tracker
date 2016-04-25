import json, logging
import numpy as np

from . import TurnTrackerSet, Vocabulary


logger = logging.getLogger(__name__)


class Dstc2(TurnTrackerSet):
    turn_marker = ('START_TURN_SYS', 'START_USR', None, None, "none none none")
    sys_usr_delim = 'DELIM'

    def __init__(self, filename, 
            max_turn_len=None, first_n=None, words_vocab=None, labels_vocab=None, sample_unk=0):
        start_turn, delim = self.__class__.turn_marker, self.__class__.sys_usr_delim
        self._raw_data = raw_data = json.load(open(filename))
        self._first_n = min(first_n, len(raw_data)) if first_n else len(raw_data)
        turns = [(turn[0] + ' %s ' % delim + turn[1]).split() for dialog in raw_data for turn in [start_turn] + dialog]
        labels = [turn[4] for dialog in raw_data for turn in [start_turn] + dialog]
        assert len(turns) == len(labels), '%s vs %s' % (turns, labels)

        turns, labels = turns[:first_n], labels[:first_n]

        self._vocab = words_vocab or Vocabulary([w for turn in turns for w in turn], extra_words=[delim])
        self._lab_vocab = labels_vocab or Vocabulary(labels)

        self._turn_lens = [len(t) for t in turns]
        s = sorted(self._turn_lens)
        max_turn, perc95 = s[-1], s[int(0.95 * len(s))]
        self._turn_len = max_turn_len = max_turn_len or max_turn
        logger.info('Turn length: %4d.\nMax turn len %4d.\n95-percentil %4d.\n', max_turn_len, max_turn, perc95)

        self._turns = np.zeros((len(turns), max_turn_len), dtype=np.int64)
        for i, t in enumerate(turns):
            for j, w in enumerate(t):
                self._turns[i, j] = self._vocab.get_i(w, unk_chance_smaller=sample_unk)
        logger.info('Labels are join slots represented as string e.g. "indian east moderate"')
        self._labels = np.array([self.labels_vocab.get_i(l) for l in labels], dtype=np.int64)

    def __len__(self):
        return self._first_n

    @property
    def turns(self):
        '''
        Returns list of fixed lenght numpy arrays encoding the
        input of a turn.
        Shorter turns are padded with zeros.
        '''
        return self._turns

    @property
    def labels(self):
        '''
        Returns list of fixed length numpy arrays
        '''
        return self._labels

    @property
    def words_vocab(self):
        return self._vocab

    @property
    def labels_vocab(self):
        return self._lab_vocab

    @property
    def turn_lens(self):
        res = np.array(self._turn_lens, dtype=np.int64)
        res.shape = (len(res))
        return res 

    @property
    def max_turn_len(self):
        return self._turn_len
