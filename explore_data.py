import json

def get_all_labels(filename):
    data = []
    with open(filename) as f:
        data = json.load(open(filename))

    label_set = set()
    label_set_separated = [set(), set(), set()]

    for dialog in data:
        for turn in dialog:
            label = turn[4].split()
            if len(label) == 4:
                fst = label.pop(0)
                label[0] = fst + ' ' + label[0]
            label_set.add(tuple(label))
            for i, l in enumerate(label):
                label_set_separated[i].add(l)

    return label_set, label_set_separated

def stat_labels_separated(labels_dev, labels_train):
    for i, lab_dev in enumerate(labels_dev):
        lab_train = labels_train[i]
        extra_in_dev = lab_dev.difference(lab_train)
        print("label no. {}: unique: {}, unseen: {}".format(i, len(lab_dev), len(extra_in_dev)))


def main():
    train_filename = './data/dstc2/data.dstc2.train.json'
    dev_filename = './data/dstc2/data.dstc2.dev.json'

    train_labels, train_labels_separated = get_all_labels(train_filename)
    dev_labels, dev_labels_separated = get_all_labels(dev_filename)

    extra_in_dev_labels = dev_labels.difference(train_labels)

    print("comparing files:\n\t{}\n\t{}".format(dev_filename, train_filename))

    print("unique triple of labels: {}".format(len(dev_labels)))
    print("unseen triple of labels: {}".format(len(extra_in_dev_labels)))

    stat_labels_separated(dev_labels_separated, train_labels_separated)


if __name__ == '__main__':
    main()