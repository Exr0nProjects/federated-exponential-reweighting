from matplotlib import pyplot as plt
import tensorflow_federated as tff
import numpy as np

from experiments import make_federated_data

EXPERIMENT = 'gaussian_1_5'
# EXPERIMENT = 'default'
CLIENT_RATIO = 0.1

NUM_CLIENTS = 32
NUM_EPOCHS = 20000
BATCH_SIZE = 32
PREFETCH_BUFFER = 10
SHUFFLE_BUFFER = 100

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
train_data = emnist_train
test_data = emnist_test

sampled_clients = np.random.choice(train_data.client_ids, NUM_CLIENTS)
dataset, _, affected_clients = make_federated_data(NUM_EPOCHS, SHUFFLE_BUFFER, BATCH_SIZE, PREFETCH_BUFFER,
                                                train_data, sampled_clients, EXPERIMENT, CLIENT_RATIO)

print(f"Of {len(dataset)} total clients, {len(affected_clients)} ({len(affected_clients)/len(dataset)*100}%) were affected")

def show_digits(client, desc):
    client_ds = list(client.take(1))[0]
    figure = plt.figure(figsize=(20, 4))
    figure.suptitle(desc)
    for i, example in enumerate(client_ds['x']):
        plt.subplot(4, 8, i+1)
        plt.imshow(example.numpy(), cmap='gray', aspect='equal')
        plt.title(f"label: {client_ds['y'][i][0]}")
        plt.axis('off')
    plt.show()

show_digits(affected_clients[0], EXPERIMENT)
