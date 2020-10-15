import numpy as np


class EarlyStopping:
    """
    The source code of this class is under the MIT License and was copied from
    the Keras project, and has been modified.

    Stop training when a monitored quantity has stopped improving.

    Args:
        min_delta (float): Minimum change in the monitored quantity to qualify
            as an improvement, i.e. an absolute change of less than min_delta,
            will count as no improvement. (Default value = 0)
        patience (int): Number of epochs with no improvement after which
            training will be stopped. (Default value = 0)
        verbose (bool): Whether to print when early stopping is done.
            (Default value = False)
        mode (string): One of {min, max}. In `min` mode, training will stop when
            the quantity monitored has stopped decreasing; in `max` mode it will
            stop when the quantity monitored has stopped increasing.
            (Default value = 'min')
    """

    def __init__(self, min_delta=0, patience=0, verbose=False, mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

        if mode not in ['min', 'max']:
            raise ValueError("Invalid mode '%s'" % mode)
        self.mode = mode

        if mode == 'min':
            self.min_delta *= -1
            self.monitor_op = np.less
        elif mode == 'max':
            self.min_delta *= 1
            self.monitor_op = np.greater

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.mode == 'min' else -np.Inf

    def on_epoch_end(self, epoch, logs):
        current = logs
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True

    def on_train_end(self):
        if self.stopped_epoch > 0 and self.verbose:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
