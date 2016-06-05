Dialogue State Tracker
======================

Josef Valek, June 2016

I have taken model created by Petr Belohlavek and Vojtech Hudecek (referred to as `cool_model`) and tweaked it a bit to predict labels of dialogue state
separately instead of jointly. I will only focus on differences of the new model vs the old one and a bit of reasoning why this approach could work better.

Differences with `cool_model`
-----------------------------

`separated_model` is essentially the same as `cool_model`. The only (but important) difference is that labels are not predicted jointly,
but instead each is predicted separately (using separate logits - and probabilities, but based on the same hidden state).
Loss is sum of losses on individual labels (those being calculated in the same way as in `cool_model` but each individually).

Why to predict labels separately?
---------------------------------

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

shows that this could make a difference, because new model could possibly predict any of the `66*7*5=2310` labels whereas the old one
even if trained to be as accurate as possible can only make 325 different labels (ie. only exactly those seen in training data).

Note on Accuracy
----------------

Accuracy of the model is measured in the same way as in `cool_model`, ie. label is predicted as triple and only considered
correct if all members of the triple are correct. Other way could be to measure "partial" accuracy (ie. 0/3, 1/3, ... 3/3).


Results
-------

Created model seems to generalize better than the previous one. In my setting it had low variance but was higly biased. With hidden state dimension
of 20, word embedding dimension 30 and batch size 1 I got after 8 epochs following results:

|Train Accuracy|Valid Accuracy|Test Accuracy|
|--------------|--------------|-------------|
|0.399669      |0.393864      |0.412568     |

In following epochs, all accuracies oscilated around 40% and did not get significantly better or worse.


Possible Improvements
---------------------

Because model is not able to achieve higher accuracy on training data, I suppose that using larger hidden state and word embedding dimensions
should improve results at least a bit, but having those too large (eg. 200 and 300) leads to high variance of the model (training and valid accuracies
differ a lot).
