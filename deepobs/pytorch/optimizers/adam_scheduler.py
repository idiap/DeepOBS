# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
# Florian Mai <florian.mai@idiap.ch>


import torch

from .scheduler_util import make_lr_schedule


class AdamDecay(torch.optim.Adam):
    max_iter = 0

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, power=0.):
        self.power = power
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.lr_sched = make_lr_schedule(self, power, self.max_iter)

    def step(self, epoch=None, closure=None):
        if epoch is None:
            super().step(closure)
        else:
            self.lr_sched.step(epoch)

    @classmethod
    def set_max_epochs(self, max_iter):
        self.max_iter = max_iter
