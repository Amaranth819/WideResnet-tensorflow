import tensorflow as tf
import numpy as np
import config

# cifar10
def unpickle_cifar10(path):
    import cPickle
    with open(path, 'rb') as f:
        dic = cPickle.load(f)
        return dic

# dic = unpickle_cifar10("../../../dataset/cifar/cifar-10-batches-py/data_batch_1")
# print(dic['data'].shape) # (10000, 3072)
# print(len(dic['labels'])) # 10000

def create_dataset(data, label, bs, repeat_size = None):
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(len(label))
    dataset = dataset.map(map_batch)
    if repeat_size is None:
        dataset = dataset.batch(bs).repeat()
    else:
        dataset = dataset.batch(bs).repeat(repeat_size)
    return dataset


def map_batch(data_batch, label_batch):
    data = tf.reshape(data_batch, config.image_size)
    label = tf.one_hot(label_batch, config.num_classes)
    return data, label

def get_next(dataset):
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()