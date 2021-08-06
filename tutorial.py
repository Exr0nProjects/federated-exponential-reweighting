import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import wandb
from matplotlib import pyplot as plt
from tqdm import trange

import collections
from functools import partial
from datetime import datetime
from time import time
import gc

from experiments import preprocess, make_federated_data

import nest_asyncio
nest_asyncio.apply()

np.random.seed(0)
wandb_run = wandb.init(entity='federated-reweighting', project='emnist-vanilla')
print('running', wandb_run.name)

EXPERIMENT = 'default'
CLIENT_RATIO = 0.1  # ratio of clients affected by noise

# NUM_MEGAPOCHS = wandb_run.config.central_epochs
# NUM_CLIENTS =   wandb_run.config.central_batch
# CENTRAL_LR =    wandb_run.config.central_lr
#
# NUM_EPOCHS =    wandb_run.config.client_epochs
# BATCH_SIZE =    wandb_run.config.client_batch
# CLIENT_LR =     wandb_run.config.client_lr

NUM_CLIENTS = 50         # number of clients to sample on each round
NUM_EPOCHS = 100         # number of times to train for each selected client subset
NUM_MEGAPOCHS = 200      # number of times to reselect clients
BATCH_SIZE = 32

CLIENT_LR = 0.001
CENTRAL_LR = 0.005

PREFETCH_BUFFER = BATCH_SIZE
SHUFFLE_BUFFER = 100

NUM_TEST_CLIENTS = 50

IMG_WIDTH = 28
IMG_HEIGHT = 28

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

train_data = emnist_train
test_data = emnist_test

example_dataset = train_data.create_tf_dataset_for_client(train_data.client_ids[0])

preprocessed_example_dataset = preprocess(NUM_EPOCHS, SHUFFLE_BUFFER, BATCH_SIZE, PREFETCH_BUFFER, example_dataset)

def create_keras_model():
    return tf.keras.models.Sequential([
        # tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Conv2D(16, (7,)*2, input_shape=(IMG_WIDTH, IMG_HEIGHT, 1), activation='relu'),
        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(300, kernel_initializer='zeros'),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])
def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

federated_eval = tff.learning.build_federated_evaluation(model_fn)  # https://stackoverflow.com/a/56811627/10372825

state = iterative_process.initialize()

seen_ids = set()
print(f"total clients: {len(train_data.client_ids)} train, {len(test_data.client_ids)} test")

for round_num in trange(0, NUM_MEGAPOCHS):
    try:
        start_time = time()
        sampled_clients = np.random.choice(train_data.client_ids, NUM_CLIENTS)
        for client in sampled_clients:
            seen_ids.add(client)
        ds, _, affected_clients = make_federated_data(NUM_EPOCHS, SHUFFLE_BUFFER, BATCH_SIZE, PREFETCH_BUFFER,
                                                      train_data, sampled_clients, EXPERIMENT, CLIENT_RATIO)
        sampled_eval_clients = np.random.choice(test_data.client_ids, NUM_TEST_CLIENTS)
        eval_dataset, _, _ = make_federated_data(NUM_EPOCHS, SHUFFLE_BUFFER, BATCH_SIZE, PREFETCH_BUFFER,
                                                 test_data, sampled_eval_clients, 'default', 0)
        state, metrics = iterative_process.next(state, ds)
        gc.collect()
        eval_metrics = federated_eval(state.model, eval_dataset)
        # print('round {:2d}, metrics={}, evalmetrics={}'.format(round_num+1, metrics['train'], eval_metrics))
        wandb.log({
            **metrics['train'],
            'step': round_num * NUM_EPOCHS,
            'test_accuracy': eval_metrics['sparse_categorical_accuracy'],
            'test_loss': eval_metrics['loss'],
            'client_coverage': len(seen_ids)/len(train_data.client_ids),
            'time_taken': time() - start_time
        })
        # if round_num % 30 == 29:
        #     fcm.save_checkpoint(state, round_num+1)
        #     print('saved checkpoint', run.name, round_num+1)
    except KeyboardInterrupt:
        print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\ninterrupted at', datetime.now().strftime("%T"))
        input('press enter to continue...')


