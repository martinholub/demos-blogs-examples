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

def clipped_masked_error(mask):
    def clipped_error(y_true, y_pred):
        loss = huber_loss(y_true, y_pred)
        loss *= mask
        # loss = K.dot(loss, mask)
        return K.sum(loss, axis = -1)
    return clipped_error

# def clipped_masked_error(y_true, y_pred, mask):
#     loss = huber_loss(y_true, y_pred)
#     loss *= mask
#     return K.sum(loss, axis = -1)

# Metrics
def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=1)) # TODO: previously was axis=-1. Why??

# This will probably not work, bettter to do it with callback
def get_adam_lr_metric(optimizer):
    """Gets Adam's lr_t value """
    def lr_metric(y_true = [], y_pred = []):
        # Get vals
        decay = optimizer.decay
        lr = optimizer.lr
        iters = optimizer.iterations # only this should not be const
        beta_1 = optimizer.beta_1
        beta_2 = optimizer.beta_2
        # calculate
        lr = lr * (1. / (1. + decay * K.cast(iters, K.dtype(decay))))
        t = K.cast(iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(beta_2, t)) / (1. - K.pow(beta_1, t)))
        return lr_t
    return lr_metric
