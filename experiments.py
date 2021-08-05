from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial
from collections import OrderedDict
from random import random

def assign(ordered_dict, values):
    return OrderedDict(list(ordered_dict.items()) + list(values.items()))

def gausian_blur(data_ratio, size, ds):
    def blur_ex(ex):
        if random() < data_ratio:
            return assign(ex, { 'pixels': tfa.image.gaussian_filter2d(ex['pixels'], filter_shape=size) })
        else:
            return ex

    return ds.map(blur_ex)

experiments = {
    'default':          lambda ds: ds,
    'gaussian_1_3': partial(gausian_blur, 1., 1),
    'gaussian_1_5': partial(gausian_blur, 1., 4)
}

def preprocess(num_epochs, shuffle_buffer, batch_size, prefetch_buffer, dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return OrderedDict(
            x=tf.expand_dims(element['pixels'], -1),
            y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(num_epochs).shuffle(shuffle_buffer).batch(batch_size
                    ).map(batch_format_fn).prefetch(prefetch_buffer)

def make_federated_data(num_epochs, shuffle_buffer, batch_size, prefetch_buffer, client_data, client_ids, experiment, client_ratio):
    full = []
    normal = []
    affected = []
    for id in client_ids:
        ds = client_data.create_tf_dataset_for_client(id)
        rand = random()
        if rand < client_ratio:
            ds = experiments[experiment](ds)
        ds = preprocess(num_epochs, shuffle_buffer, batch_size, prefetch_buffer, ds)
        if rand < client_ratio:
            affected.append(ds)
        else:
            normal.append(ds)
        full.append(ds)
    return full, normal, affected
