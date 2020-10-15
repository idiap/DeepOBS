# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
# Florian Mai <florian.mai@idiap.ch>

from torch.optim import lr_scheduler


def make_lr_schedule(optimizer, power=None, max_epoch=None):
    if power is None or max_epoch is None:
        def determine_lr(epoch): return 1

    else:
        def determine_lr(epoch):
            print('LR changed by {}'.format((1. - epoch/max_epoch) ** power))
            return (1. - epoch/max_epoch) ** power

    sched = lr_scheduler.LambdaLR(optimizer, determine_lr)
    return sched
