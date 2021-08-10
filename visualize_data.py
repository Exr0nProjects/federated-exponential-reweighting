from matplotlib import pyplot as plt
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
from random import shuffle

from experiments import make_federated_data, experiments

# EXPERIMENT = 'gaussian_1_5'
# EXPERIMENT = 'shift_label_up'
# EXPERIMENT = 'default'
CLIENT_RATIO = 0.9

NUM_CLIENTS = 32
NUM_EPOCHS = 20000
BATCH_SIZE = 1280
PREFETCH_BUFFER = 10
SHUFFLE_BUFFER = 100

WIDTH = 64
HEIGHT = 48

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
train_data = emnist_train
test_data = emnist_test

def show_digits(arr, desc):
    figure = plt.figure(figsize=(WIDTH, HEIGHT))
    figure.suptitle(desc)
    figure.subplots_adjust(hspace=0, wspace=0)
    for i, example in enumerate(arr['x']):
        if i == HEIGHT * WIDTH: break
        # plt.imsave(f"out/example_{experiment}_{i}.png", tf.expand_dims(example, axis=-1).numpy(), cmap='gray')
        plt.subplot(HEIGHT, WIDTH, i+1)
        plt.imshow(example.numpy(), cmap='gray', aspect='equal')
        plt.xticks([])
        plt.yticks([])
        # plt.xlabel(f"label: {client_ds['y'][i][0]}")
    # plt.show()
    plt.savefig(f"out/visualize_{desc}_{WIDTH}x{HEIGHT}.png", transparent=True, dpi=200)

def show(experiment):
    print("processing", experiment)
    sampled_clients = np.random.choice(train_data.client_ids, NUM_CLIENTS)
    dataset, _, affected_clients = make_federated_data(NUM_EPOCHS, SHUFFLE_BUFFER, BATCH_SIZE, PREFETCH_BUFFER,
                                                       train_data, sampled_clients, experiment, CLIENT_RATIO)

    print(f"Of {len(dataset)} total clients, {len(affected_clients)} ({len(affected_clients)/len(dataset)*100}%) were affected")

    return list(list(affected_clients[0].take(1))[0]['x'])

    # show_digits(list(affected_clients[0].take(1))[0], experiment)


if __name__ == '__main__':
    all_images = []
    print(len(all_images))
    for experiment in ['default', 'gaussian_1_5', 'mask_right_half', 'mask_left_half', 'mask_top_half', 'mask_bot_half']:
        all_images += show(experiment)
    shuffle(all_images)
    # show('default')
    show_digits({ 'x': all_images }, 'combo')
