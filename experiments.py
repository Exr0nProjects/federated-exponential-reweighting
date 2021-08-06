from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial
from collections import OrderedDict
from random import random

tf.compat.v1.enable_eager_execution()

def assign(ordered_dict, values):
    return OrderedDict(list(ordered_dict.items()) + list(values.items()))

def gausian_blur(data_ratio, size, ds):
    def blur_ex(ex):
        if random() < data_ratio:
            return assign(ex, { 'pixels': tfa.image.gaussian_filter2d(ex['pixels'], filter_shape=size) })
        else:
            return ex

    return ds.map(blur_ex)

def mask(y1, x1, y2, x2, ds):
    def lam(ex):
        sz = ex['pixels'].shape
        img = [x[:] for x in [[0]*sz[0]]*sz[1]]
        for y in range(sz[0]):
            for x in range(sz[1]):
                img[y][x] = ex['pixels'][y][x]
        for y in range(int(sz[0]*y1), int(sz[0]*y2)):
            for x in range(int(sz[1]*x1), int(sz[1]*x2)):
                img[y][x] = 0.5
        return assign(ex, { 'pixels': tf.convert_to_tensor(img) })

    return ds.map(lam)

def shift_label(shift, modulo, ds):
    def lam(ex):
        print(ex)
        return assign(ex, { 'label': (ex['label']+shift)%10 })
    return ds.map(lam)

def set_label(val, ds):
    def lam(ex):
        return assign(ex, { 'label': val })
    return ds.map(lam)

def swap_label(a, b, ds):
    def lam(ex):
        if ex['label'] == a:
            return assign(ex, { 'label': b })
        if ex['label'] == b:
            return assign(ex, { 'label': a })
    return ds.map(lam)

experiments = {
    'default':          lambda ds: ds,
    'gaussian_1_3':     partial(gausian_blur, 1., 1),
    'gaussian_1_5':     partial(gausian_blur, 1., 4),
    'mask_right_third': partial(mask, 0, 0.67, 1, 1),
    'mask_bot_third':   partial(mask, 0.67, 0, 1, 1),
    'mask_left_third':  partial(mask, 0, 0, 1, 0.33),
    'mask_top_third':   partial(mask, 0, 0, 0.33, 1),
    'mask_right_half':  partial(mask, 0, 0.5, 1, 1),
    'mask_bot_half': partial(mask, 0.5, 0, 1, 1),
    'mask_left_half':   partial(mask, 0, 0, 1, 0.5),
    'mask_top_half':    partial(mask, 0, 0, 0.5, 1),
    'shift_label_up':   partial(shift_label, 1, 10),
    'zero_labels':      partial(set_label, 0),
    'swap_three_seven': partial(swap_label, 3, 7),
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
