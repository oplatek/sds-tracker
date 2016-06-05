Predicting separate labels
==========================

`separated_model` is essentially the same as `cool_model`. The only difference is that labels are not predicted jointly,
but instead each is predicted separately (but based on the same hidden state). Loss is sum of losses on individual
labels (those being calculated in the same way as in `cool_model`).

It seems that this model could do better than the original one, because predicting labels separately brings possibility to
generalize more - to predict triple of labels which was not seen in the data - which is not possible in original model.

Just a glance of the data (output of `explore_data.py`)

```
comparing files:
	./data/dstc2/data.dstc2.dev.json
	./data/dstc2/data.dstc2.train.json
unique triple of labels: 325
unseen triple of labels: 41
label no. 0: unique: 66, unseen: 1
label no. 1: unique: 7, unseen: 0
label no. 2: unique: 5, unseen: 0
```

shows how this generalization is important.

Note on accuracy
----------------

Accuracy of the model is measured in the same way as in `cool_model`, ie. label is predicted as triple and only considered
correct if all members of the triple are correct. Other way could be to measure "partial" accuracy (ie. 0/3, 1/3, ... 3/3)





