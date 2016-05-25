import json, logging
import numpy as np

from . import TurnTrackerSet, Vocabulary


logger = logging.getLogger(__name__)


class Dstc2(TurnTrackerSet):
    sys_usr_delim = 'DELIM'

    def __init__(self, filename, 
            max_turn_len=None, max_dial_len=None, 
            first_n=None, words_vocab=None, labels_vocab=None, sample_unk=0):
        delim = self.__class__.sys_usr_delim
        self._raw_data = raw_data = json.load(open(filename))
        self._first_n = min(first_n, len(raw_data)) if first_n else len(raw_data)
        if max_dial_len != None:
            dialogs = [[(turn[0] + ' %s ' % delim + turn[1]).split() for turn in dialog] for dialog in raw_data if len(dialog) <= max_dial_len]
            labels = [[turn[4] for turn in dialog] for dialog in raw_data if len(dialog) <= max_dial_len]
        else:
            dialogs = [[(turn[0] + ' %s ' % delim + turn[1]).split() for turn in dialog] for dialog in raw_data]
            labels = [[turn[4] for turn in dialog] for dialog in raw_data]
        assert len(dialogs) == len(labels), '%s vs %s' % (dialogs, labels)

        dialogs, labels = dialogs[:first_n], labels[:first_n]

        self._vocab = words_vocab or Vocabulary([w for turns in dialogs for turn in turns for w in turn], extra_words=[delim])
        self._lab_vocab = labels_vocab or Vocabulary([w for turn in labels for w in turn])

        s = sorted([len(t) for turns in dialogs for t in turns])
        max_turn, perc95t = s[-1], s[int(0.95 * len(s))]
        self._max_turn_len = mtl = max_turn_len or max_turn
        logger.info('Turn length: %4d.\nMax turn len %4d.\n95-percentil %4d.\n', mtl, max_turn, perc95t)
        d = sorted([len(d) for d in dialogs])
        max_dial, perc95d = d[-1], d[int(0.95 * len(d))]
        self._max_dial_len = mdl = max_dial_len or max([len(d) for d in dialogs]) 
        logger.info('Dial length: %4d.\Dial turn len %4d.\n95-percentil %4d.\n', mdl, max_dial, perc95d)

        self._turn_lens_per_dialog = np.zeros((len(dialogs), mdl), dtype=np.int64)
        self._dials = np.zeros((len(dialogs), mdl, mtl), dtype=np.int64)
        self._dial_lens = np.array([len(d) for d in dialogs], dtype=np.int64)

        self._dial_mask = np.zeros((len(self._dials), self._max_dial_len), dtype=np.int64)
        for i, l in enumerate(self._dial_lens):
            self._dial_mask[i, :l] = np.ones((l,), dtype=np.int64)

        for i, d in enumerate(dialogs):
            for j, t in enumerate(d):
                if j > mdl or len(t) > mtl:  
                    # Keep prefix of turns, discard exceeding turns
                    # because of a) num_turns too big b) current turn too long.
                    break 
                self.turn_lens[i, j] = len(t)
                for k, w in enumerate(t):
                    self._dials[i, j, k] = self._vocab.get_i(w, unk_chance_smaller=sample_unk)

        logger.info('Labels are join slots represented as string e.g. "indian east moderate"')
        self._labels = np.zeros((len(dialogs), mdl), dtype=np.int64)
        for i, d in enumerate(labels):
            for j, turn_label in enumerate(d):
                self._labels[i, j] = self.labels_vocab.get_i(turn_label)

    def __len__(self):
        return self._first_n

    @property
    def dial_mask(self):
        return self._dial_mask

    @property
    def dialogs(self):
        '''Returns np.array with shape [ # dialogs, max_dial_len, max_turn_len]'''
        return self._dials

    @property
    def labels(self):
        '''
        '''
        return self._labels

    @property
    def words_vocab(self):
        return self._vocab

    @property
    def labels_vocab(self):
        return self._lab_vocab

    @property
    def dial_lens(self):
        '''
        :return: np.array with shape [#dialogs]
        '''
        return self._dial_lens
    @property
    def turn_lens(self):
        '''Returns np.array with shape [ # dialogs, max_dialog_len]'''
        return self._turn_lens_per_dialog

    @property
    def max_turn_len(self):
        return self._max_turn_len

    @property
    def max_dial_len(self):
        return self._max_dial_len
