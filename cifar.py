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
    # data = tf.reshape(data_batch, config.image_size)
    # data_aug = data_augmentation(data)
    data = tf.transpose(tf.reshape(data_batch, config.ori_size), [1, 2, 0])
    label = tf.one_hot(label_batch, config.num_classes)
    return data, label

def data_augmentation(img):
    tf.set_random_seed(233)
    fliplr = tf.image.random_flip_left_right(img)
    fliptb = tf.image.random_flip_up_down(fliplr)
    offset_height = np.random.randint(5)
    offset_width = np.random.randint(5)
    translation = tf.image.pad_to_bounding_box(fliptb, offset_height, offset_width, 32 + offset_height, 32 + offset_width)
    output = tf.image.crop_to_bounding_box(translation, 0, 0, 32, 32)
    return output

def get_next(dataset):
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
