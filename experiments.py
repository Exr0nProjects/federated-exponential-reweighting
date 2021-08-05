from scipy.ndimage.filters import gaussian_filter
from functools import partial
from collections import OrderedDict
from random import random

def assign(ordered_dict, values):
    return OrderedDict(ordered_dict.keys() + values.keys())

def gausian_blur(client_ratio, data_ratio, sigma, ds):
    modified = []

    def blur_ex(ex):
        if random() < client_ratio:
            print(ex)
            return assign(ex, { 'pixels': gaussian_filter(ex['pixels'], sigma=sigma) })
        else:
            return ex

    return ds.map(blur_ex), modified


experiments = {
    'default':          lambda ds: ds,
    'gaussian_10e_100': partial(gausian_blur, 10, 100)
}
