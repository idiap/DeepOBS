# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
# Florian Mai <florian.mai@idiap.ch>


import torch
from torch import nn
from .testproblems_modules import net_char_rnn
from ..datasets.tolstoi import tolstoi
from .testproblem import UnregularizedTestproblem


class tolstoi_char_rnn(UnregularizedTestproblem):
    def __init__(self, batch_size, weight_decay=None):
        """Create a new Char RNN test problem instance on Tolstoi.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(tolstoi_char_rnn, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model."
            )

    def set_up(self):
        """Set up the Wide ResNet 16-4 test problem on SVHN."""
        seq_length = 50
        vocab_size = 83  # For War and Peace
        num_layers = 2
        hidden_dim = 128
        self.data = tolstoi(self._batch_size, seq_length=seq_length)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_char_rnn(seq_len=seq_length, hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=num_layers)
        self.net.to(self._device)
        
        # make sure we're not regularizing
        self.regularization_groups = self.get_regularization_groups()
        
    def _get_next_batch(self):
        X, Y = UnregularizedTestproblem._get_next_batch(self)
        return X, Y.view(-1)
