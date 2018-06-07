from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList

# Losses
def huber_loss(y_true, y_pred, clip_value = 1.0):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if hasattr(tf, 'select'):
        return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
    else:
        return tf.where(condition, squared_loss, linear_loss)  # condition, true, false


def clipped_error(y_true, y_pred):
    return K.mean(huber_loss(y_true, y_pred), axis=-1)

# Metrics
def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

# Callbacks
class Callback(KerasCallback):
    def _set_env(self, env):
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        """Called at beginning of each episode"""
        pass

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        pass

    def on_step_begin(self, step, logs={}):
        """Called at beginning of each step"""
        pass

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        pass

    def on_action_begin(self, action, logs={}):
        """Called at beginning of each action"""
        pass

    def on_action_end(self, action, logs={}):
        """Called at end of each action"""
        pass

class FileLogger(Callback):
    def __init__(self, filepath, interval=None):
        self.filepath = filepath
        self.interval = interval

        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dict that maps from episode to metrics array.
        self.metrics = {}
        self.starts = {}
        self.data = {}

    def on_train_begin(self, logs):
        """ Initialize model metrics before training """
        self.metrics_names = self.model.metrics_names

    def on_train_end(self, logs):
        """ Save model at the end of training """
        self.save_data()

    def on_episode_begin(self, episode, logs):
        """ Initialize metrics at the beginning of each episode """
        assert episode not in self.metrics
        assert episode not in self.starts
        self.metrics[episode] = []
        self.starts[episode] = timeit.default_timer()

    def on_episode_end(self, episode, logs):
        """ Compute and print metrics at the end of each episode """
        duration = timeit.default_timer() - self.starts[episode]

        metrics = self.metrics[episode]
        if np.isnan(metrics).all():
            mean_metrics = np.array([np.nan for _ in self.metrics_names])
        else:
            mean_metrics = np.nanmean(metrics, axis=0)
        assert len(mean_metrics) == len(self.metrics_names)

        data = list(zip(self.metrics_names, mean_metrics))
        data += list(logs.items())
        data += [('episode', episode), ('duration', duration)]
        for key, value in data:
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

        if self.interval is not None and episode % self.interval == 0:
            self.save_data()

        # Clean up.
        del self.metrics[episode]
        del self.starts[episode]

    def on_step_end(self, step, logs):
        """ Append metric at the end of each step """
        self.metrics[logs['episode']].append(logs['metrics'])

    def save_data(self):
        """ Save metrics in a json file """
        if len(self.data.keys()) == 0:
            return

        # Sort everything by episode.
        assert 'episode' in self.data
        sorted_indexes = np.argsort(self.data['episode'])
        sorted_data = {}
        for key, values in self.data.items():
            assert len(self.data[key]) == len(sorted_indexes)
            # We convert to np.array() and then to list to convert from np datatypes to native datatypes.
            # This is necessary because json.dump cannot handle np.float32, for example.
            sorted_data[key] = np.array([self.data[key][idx] for idx in sorted_indexes]).tolist()

        # Overwrite already open file. We can simply seek to the beginning since the file will
        # grow strictly monotonously.
        with open(self.filepath, 'w') as f:
            json.dump(sorted_data, f)
