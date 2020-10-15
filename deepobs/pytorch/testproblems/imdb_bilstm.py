# -*- coding: utf-8 -*-
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
# Florian Mai <florian.mai@idiap.ch>
# Thijs Vogels <thijs.vogels@epfl.ch>

from torch import nn
from .testproblems_modules import net_imdb_bilstm
from ..datasets.imdb import imdb
from .testproblem import TestProblem


class imdb_bilstm(TestProblem):
    """DeepOBS test problem class for a Bi-LSTM on IMDB.

  Args:
      batch_size (int): Batch size to use.
      weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
          is used on the weights but not the biases. Defaults to ``0.002``.

  Attributes:
    data: The DeepOBS data set class for IMDB.
    loss_function: The loss function for this testproblem is torch.nn.CrossEntropyLoss()
    net: The DeepOBS subclass of torch.nn.Module that is trained for this testproblem (net_imdb_bilstm).

  Methods:
      get_regularization_loss: Returns the current regularization loss of the network state.
  """

    def __init__(self, batch_size, weight_decay=0.002):
        """Create a new 3c3d test problem instance on Cifar-10.

        Args:
            batch_size (int): Batch size to use.
            weight_decay (float): Weight decay factor. Weight decay (L2-regularization)
                is used on the weights but not the biases. Defaults to ``0.002``.
        """

        super(imdb_bilstm, self).__init__(batch_size, weight_decay)

    def set_up(self):
        """Set up the LSTM test problem on Cifar-10."""
        self.data = imdb(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = net_imdb_bilstm(num_outputs=2)
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

    def get_regularization_groups(self):
        """Creates regularization groups for the parameters.

        Returns:
            dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
        """
        no, l2 = 0.0, self._weight_decay
        group_dict = {no: [], l2: []}

        for parameters_name, parameters in self.net.named_parameters():
            # penalize only the non bias layer parameters
            if 'bias' not in parameters_name:
                group_dict[l2].append(parameters)
            else:
                group_dict[no].append(parameters)
        return group_dict
