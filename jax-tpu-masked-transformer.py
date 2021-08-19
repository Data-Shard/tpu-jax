import jax
import jax.numpy as jnp               # JAX NumPy

from flax import linen as nn          # The Linen API
from flax.training import train_state
import optax                          # The Optax gradient processing and optimization library

import numpy as np                    # Ordinary NumPy
import tensorflow_datasets as tfds    # TFDS for MNIST

class CNN(nn.Module):

  @nn.compact
  # Provide a constructor to register a new parameter 
  # and return its initial value
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1)) # Flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)    # There are 10 classes in MNIST
    return x


def compute_metrics(logits, labels):
  loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes=10)))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy
  }
  return metrics


def get_datasets():
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  # Split into training/test sets
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  # Convert to floating-points
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
  return train_ds, test_ds

@jax.jit
def train_step(state, batch):
  def loss_fn(params):
    logits = CNN().apply({'params': params}, batch['image'])
    loss = jnp.mean(optax.softmax_cross_entropy(
        logits=logits, 
        labels=jax.nn.one_hot(batch['label'], num_classes=10)))
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits, batch['label'])
  return state, metrics

@jax.jit
def eval_step(params, batch):
  logits = CNN().apply({'params': params}, batch['image'])
  return compute_metrics(logits, batch['label'])

def train_epoch(state, train_ds, batch_size, epoch, rng):
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  batch_metrics = []

  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, metrics = train_step(state, batch)
    batch_metrics.append(metrics)

  training_batch_metrics = jax.device_get(batch_metrics)
  training_epoch_metrics = {
      k: np.mean([metrics[k] for metrics in training_batch_metrics])
      for k in training_batch_metrics[0]}

  print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))

  return state, training_epoch_metrics


def eval_model(model, test_ds):
  metrics = eval_step(model, test_ds)
  metrics = jax.device_get(metrics)
  eval_summary = jax.tree_map(lambda x: x.item(), metrics)
  return eval_summary['loss'], eval_summary['accuracy']


train_ds, test_ds = get_datasets()
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
cnn = CNN()
params = cnn.init(init_rng, jnp.ones([1, 28, 28, 1]))['params']
nesterov_momentum = 0.9
learning_rate = 0.001
tx = optax.sgd(learning_rate=learning_rate, nesterov=nesterov_momentum)
state = train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)
num_epochs = 10
batch_size = 32
for epoch in range(1, num_epochs + 1):
  # Use a separate PRNG key to permute image data during shuffling
  rng, input_rng = jax.random.split(rng)
  # Run an optimization step over a training batch
  state, train_metrics = train_epoch(state, train_ds, batch_size, epoch, input_rng)
  # Evaluate on the test set after each training epoch
  test_loss, test_accuracy = eval_model(state.params, test_ds)
  print('Testing - epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))
